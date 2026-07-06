"""
infer.py -- unified runtime for chatting with a saved model.

Loads a checkpoint, detects which architecture it is (the proprietary "ax0"
retention model with the AX0Tok chat protocol, or the legacy char-level "gpt"
baseline), and exposes a single streaming chat interface used by both the CLI
(chat.py) and the web app (serve.py).

    bot = Chatbot("ax0_pip.npz")
    for piece in bot.stream(history_turns, "hello", temp=0.7, topk=30):
        print(piece, end="", flush=True)

`history_turns` is a list of (role, text) with role in {"user", "pip"}.
"""

import numpy as np


def _flush_utf8(buf):
    """Split a bytes buffer into (decodable_prefix_str, leftover_bytes)."""
    try:
        return buf.decode("utf-8"), b""
    except UnicodeDecodeError as e:
        return buf[:e.start].decode("utf-8"), buf[e.start:]


class Chatbot:
    def __init__(self, path):
        d = np.load(path, allow_pickle=True)
        self.arch = str(d["arch"]) if "arch" in d.files else "gpt"
        if self.arch in ("ax0", "ax1"):
            from ax0 import load_ax0
            from recurrent import RecurrentAX0
            self.model, self.tok = load_ax0(path)
            self.rec = RecurrentAX0(self.model)   # O(1)-per-token streaming
        else:
            from chat import Loaded
            self.legacy = Loaded(path)
            self.model = self.legacy.model
        self.cfg = self.model.cfg
        self.codename = f"ax0-pip-{self.model.num_params()/1e6:.1f}m-chat"

    # -- sampling helper ---------------------------------------------------
    def _sample(self, logits_row, temp, topk, ban=()):
        logit = logits_row.astype(np.float64) / max(temp, 1e-6)
        for b in ban:
            logit[b] = -np.inf
        if topk:
            kth = np.sort(logit)[-min(topk, len(logit))]
            logit[logit < kth] = -np.inf
        logit -= logit.max()
        p = np.exp(logit); p /= p.sum()
        return int(np.random.choice(len(p), p=p))

    # -- streaming chat ----------------------------------------------------
    @np.errstate(over="ignore", invalid="ignore")
    def stream(self, history_turns, message, temp=0.7, topk=30, max_new=200):
        if self.arch in ("ax0", "ax1"):
            yield from self._stream_ax0(history_turns, message, temp, topk, max_new)
        else:
            yield from self._stream_gpt(history_turns, message, temp, topk, max_new)

    def _stream_ax0(self, history_turns, message, temp, topk, max_new):
        # uses the recurrent O(1)-state path: prime on the prompt, then step
        # one token at a time (constant work/memory, unbounded context).
        tok = self.tok
        ids = tok.encode_prompt(list(history_turns) + [("user", message)])
        ban = (tok.USER, tok.PIP, tok.PAD)   # never emit control tokens mid-reply
        self.rec.reset()
        logits = None
        for t in ids:
            logits = self.rec.step(t)
        buf = b""
        for _ in range(max_new):
            nxt = self._sample(logits, temp, topk, ban=ban)
            if nxt == tok.END:
                break
            buf += tok.vocab_bytes.get(nxt, b"")
            text, buf = _flush_utf8(buf)
            if text:
                yield text
            logits = self.rec.step(nxt)
        if buf:
            yield buf.decode("utf-8", errors="replace")

    def _stream_gpt(self, history_turns, message, temp, topk, max_new):
        L = self.legacy
        hist = ""
        for role, text in history_turns:
            hist += (f"U: {text}\n" if role == "user" else f"P: {text}\n")
        ids = L.encode(hist + f"U: {message}\nP:")
        bs = self.cfg.block_size
        nl = L.stoi.get("\n")
        started = False
        for _ in range(max_new):
            ctx = np.array([ids[-bs:]], dtype=np.int64)
            logits, _ = self.model.forward(ctx)
            nxt = self._sample(logits.data[0, -1], temp, topk)
            if nxt == nl:
                break
            ids.append(nxt)
            ch = L.itos[nxt]
            if not started and ch == " ":
                continue
            started = True
            yield ch

    def reply(self, history_turns, message, temp=0.7, topk=30):
        return "".join(self.stream(history_turns, message, temp, topk)).strip()
