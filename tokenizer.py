"""
tokenizer.py -- AX0Tok, the homemade tokenizer for ax0-pip.

Built from scratch (no sentencepiece, no tiktoken, no huggingface). Two custom
pieces:

  1. Byte-level BPE trained on OUR corpus.
     - Every input is UTF-8 bytes, so there is no unknown token, ever.
     - Common chunks (" you", "hello", "i'm", "pip") get merged into single
       tokens, so sequences get shorter -> more conversation fits in the model's
       context window, and generating a reply costs fewer steps. Smaller + cheaper.
     - Merges never cross a pre-tokenization boundary (a light word / number /
       punctuation / space regex), so we never glue across spaces/punctuation.

  2. A homemade chat protocol with real control tokens.
     Instead of the literal strings "U: " / "P: " / newline, conversations are
     framed with three special tokens:  <user>  <pip>  <end>
     The model learns turn-taking from these directly, and at chat time we just
     append <pip> and generate until <end>. Think of it as a tiny private ChatML.

Serialization is plain JSON, embedded into the model checkpoint so a model file
is fully self-describing.
"""

import json
import re
from collections import Counter

# Light pre-tokenization: contractions, letter runs, digit runs, punctuation
# runs, and whitespace -- each with an optional leading space.
_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+""")

SPECIALS = ["<pad>", "<user>", "<pip>", "<end>"]


def _pretokenize(text):
    return _PAT.findall(text)


class AX0Tok:
    def __init__(self, merges, pattern_specials=SPECIALS):
        # merges: list of (a, b) pairs, in merge order; child ids -> 256+index
        self.merges = [tuple(m) for m in merges]
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.merge_to_id = {pair: 256 + i for i, pair in enumerate(self.merges)}

        # build id -> bytes table
        self.vocab_bytes = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab_bytes[256 + i] = self.vocab_bytes[a] + self.vocab_bytes[b]

        # specials live at the very top of the id range
        base = 256 + len(self.merges)
        self.specials = {name: base + i for i, name in enumerate(pattern_specials)}
        self.special_names = pattern_specials
        for name, idx in self.specials.items():
            self.vocab_bytes[idx] = b""           # render as nothing

        self.USER = self.specials["<user>"]
        self.PIP = self.specials["<pip>"]
        self.END = self.specials["<end>"]
        self.PAD = self.specials["<pad>"]

    @property
    def vocab_size(self):
        return 256 + len(self.merges) + len(self.specials)

    # -- training ----------------------------------------------------------
    @classmethod
    def train(cls, texts, vocab_size=512, verbose=False):
        n_merges = max(0, vocab_size - 256 - len(SPECIALS))
        # word = a pre-token chunk as a tuple of byte ids; count frequencies
        words = Counter()
        for t in texts:
            for chunk in _pretokenize(t):
                words[tuple(chunk.encode("utf-8"))] += 1

        merges = []
        words = dict(words)
        for step in range(n_merges):
            pairs = Counter()
            for w, freq in words.items():
                for i in range(len(w) - 1):
                    pairs[(w[i], w[i + 1])] += freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            if pairs[best] < 2:       # nothing worth merging anymore
                break
            new_id = 256 + len(merges)
            merges.append(best)
            words = {cls._merge_word(w, best, new_id): f for w, f in words.items()}
            if verbose and step % 64 == 0:
                print(f"  merge {step:4d}: {best} (count {pairs[best]})")
        return cls(merges)

    @staticmethod
    def _merge_word(w, pair, new_id):
        out, i = [], 0
        while i < len(w):
            if i < len(w) - 1 and (w[i], w[i + 1]) == pair:
                out.append(new_id); i += 2
            else:
                out.append(w[i]); i += 1
        return tuple(out)

    # -- encode / decode ---------------------------------------------------
    def _encode_chunk(self, ids):
        while len(ids) >= 2:
            best_rank, best_i = None, None
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            a, b = ids[best_i], ids[best_i + 1]
            ids = ids[:best_i] + [self.merge_to_id[(a, b)]] + ids[best_i + 2:]
        return ids

    def encode(self, text):
        out = []
        for chunk in _pretokenize(text):
            out.extend(self._encode_chunk(list(chunk.encode("utf-8"))))
        return out

    def decode(self, ids):
        return b"".join(self.vocab_bytes.get(i, b"") for i in ids).decode(
            "utf-8", errors="replace")

    # -- chat protocol -----------------------------------------------------
    def encode_turn(self, role, text):
        """role in {'user','pip'} -> [<role>] + tokens + [<end>]."""
        head = self.USER if role == "user" else self.PIP
        return [head] + self.encode(text) + [self.END]

    def encode_dialogue(self, turns):
        ids = []
        for role, text in turns:
            ids.extend(self.encode_turn(role, text))
        return ids

    def encode_prompt(self, turns):
        """Encode history, then open a Pip turn for generation."""
        return self.encode_dialogue(turns) + [self.PIP]

    # -- serialization -----------------------------------------------------
    def to_json(self):
        return json.dumps({"merges": self.merges, "specials": self.special_names})

    @classmethod
    def from_json(cls, s):
        d = json.loads(s)
        return cls(d["merges"], d.get("specials", SPECIALS))


if __name__ == "__main__":
    import sys
    from data import build_corpus
    sample = build_corpus()
    # train on raw lines (strip the U:/P: prefixes -> just message text)
    texts = []
    for line in sample.splitlines():
        if line[:3] in ("U: ", "P: "):
            texts.append(line[3:])
    tok = AX0Tok.train(texts, vocab_size=512, verbose=True)
    print(f"\nvocab size: {tok.vocab_size}  (256 bytes + {len(tok.merges)} merges + {len(tok.specials)} specials)")
    probe = "hello pip, how are you? i'm doing great!"
    ids = tok.encode(probe)
    print(f"sample text : {probe!r}")
    print(f"encoded ids : {ids}")
    print(f"n tokens    : {len(ids)}  vs {len(probe)} chars  -> {len(probe)/len(ids):.2f} chars/token")
    print(f"roundtrip   : {tok.decode(ids)!r}  ok={tok.decode(ids)==probe}")
    # show a few learned merges as text
    learned = [tok.decode([256 + i]) for i in range(min(20, len(tok.merges)))]
    print(f"first merges: {learned}")
