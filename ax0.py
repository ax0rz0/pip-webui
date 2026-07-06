"""
ax0.py -- the AX0 / AX1 model family. Proprietary, retention-based, linear-time
decoders built from scratch on engine.Tensor and trained on CPU in numpy.

This is NOT a vanilla GPT and uses NO downloaded weights. Each block:

  1. Token-shift  -- learnable per-channel blend of each token with the previous
     one (cheap local context).
  2. Phasor retention  -- attention's softmax replaced by a per-head complex
     decay weighting on the scores:
            A[t,s] = (q_t . k_s)/sqrt(d) * r_h^(t-s) * cos(theta_h * (t-s)) ,  s<=t
     Two head families share the block:
        * memory heads     (theta = 0): pure exponential decay at many scales
                                        -- this is the AX0 mixer.
        * resonator heads  (theta > 0): decaying COSINE at many frequencies
                                        -- the AX1 addition. Captures position
                                        and periodic structure directly in the
                                        mixer, so there are NO positional embeddings.
  3. Swish-gated retention output + GeGLU feed-forward (expressive gating for a
     tiny model), with residual dropout.

The decay-only model (mixer="decay") is "AX0"; the decay+resonator model
(mixer="phasor") is "AX1". They are the same code; the only difference is the
per-head (r, theta) schedule, which is stored in the checkpoint.

Why it serves cheaply: the parallel form above is a T x T matmul (used for
training), but the same recurrence has an O(1)-state recurrent form
    S_t = (r e^{i theta}) S_{t-1} + k_t^T v_t,  o_t = Re(q_t S_t)
i.e. constant memory per token at inference. Small, cheap, fast.

Honest prior art: decay / complex-decay linear attention exists (RetNet, GLA,
RWKV, diagonal complex SSMs). AX1's specific real-valued multi-frequency
"memory + resonator" head split, plus token-shift and swish/GeGLU gating and no
positions, is our own recipe, implemented here from scratch.
"""

import gc

import numpy as np
from engine import Tensor, embedding, layernorm, cross_entropy
from optim import AdamW


class AX0Config:
    def __init__(self, vocab_size, block_size=128, n_layer=6, n_head=4,
                 n_embd=192, ff_mult=3, mixer="decay", dropout=0.0,
                 decay_lo=0.10, decay_hi=0.001, res_decay=0.98,
                 freq_lo=3.0, freq_hi=None, seed=1337,
                 chunked=False, chunk_size=64):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.ff_mult = ff_mult
        self.chunked = chunked        # O(T) chunked retention for training
        self.chunk_size = chunk_size
        self.mixer = mixer            # "decay" (AX0) or "phasor" (AX1)
        self.dropout = dropout
        self.decay_lo = decay_lo      # memory-head decay range (1-lo .. 1-hi)
        self.decay_hi = decay_hi
        self.res_decay = res_decay    # resonator-head decay envelope
        self.freq_lo = freq_lo        # shortest resonator period
        self.freq_hi = freq_hi        # longest period (defaults to block_size)
        self.seed = seed


def _p(shape, std):
    return Tensor(np.random.randn(*shape).astype(np.float32) * std)


class AX0:
    def __init__(self, cfg):
        self.cfg = cfg
        np.random.seed(cfg.seed)
        C, V, L, H = cfg.n_embd, cfg.vocab_size, cfg.n_layer, cfg.n_head
        T, Hff = cfg.block_size, cfg.ff_mult * cfg.n_embd
        assert C % H == 0, "n_embd must be divisible by n_head"
        self.head_dim = C // H
        self.dropout = cfg.dropout
        self.training = True
        self.mixer = cfg.mixer
        self.arch = "ax1" if cfg.mixer == "phasor" else "ax0"
        self.chunked = getattr(cfg, "chunked", False)
        self.chunk_size = getattr(cfg, "chunk_size", 64)
        std = 0.02
        res_std = 0.02 / np.sqrt(2 * L)

        self.tok_emb = _p((V, C), std)        # no positional embedding

        self.blocks = []
        for _ in range(L):
            self.blocks.append({
                "ln1_g": Tensor(np.ones(C, np.float32)),
                "ln1_b": Tensor(np.zeros(C, np.float32)),
                "mix":   Tensor(np.zeros(C, np.float32)),
                "Wq": _p((C, C), std), "Wk": _p((C, C), std), "Wv": _p((C, C), std),
                "Wg": _p((C, C), std),
                "rn_g": Tensor(np.ones(C, np.float32)),
                "rn_b": Tensor(np.zeros(C, np.float32)),
                "Wo": _p((C, C), res_std),
                "ln2_g": Tensor(np.ones(C, np.float32)),
                "ln2_b": Tensor(np.zeros(C, np.float32)),
                "Wu": _p((C, Hff), std), "Wv2": _p((C, Hff), std),
                "Wd": _p((Hff, C), res_std),
            })

        self.lnf_g = Tensor(np.ones(C, np.float32))
        self.lnf_b = Tensor(np.zeros(C, np.float32))

        r, theta = self._make_schedule(H, T)
        self.set_schedule(r, theta)
        self._shift = np.eye(T, k=-1, dtype=np.float32)   # token-shift matrix
        self.scale = 1.0 / np.sqrt(self.head_dim)

    # -- per-head (r, theta) schedule -------------------------------------
    def _make_schedule(self, H, T):
        cfg = self.cfg
        if cfg.mixer == "phasor":
            n_mem = max(1, H // 2)
            n_osc = H - n_mem
            r_mem = 1.0 - np.logspace(np.log10(cfg.decay_lo),
                                      np.log10(cfg.decay_hi), n_mem)
            th_mem = np.zeros(n_mem)
            if n_osc > 0:
                hi = cfg.freq_hi or T
                periods = np.geomspace(cfg.freq_lo, hi, n_osc)
                th_osc = 2 * np.pi / periods
                r_osc = np.full(n_osc, cfg.res_decay)
            else:
                th_osc = np.zeros(0); r_osc = np.zeros(0)
            r = np.concatenate([r_mem, r_osc])
            theta = np.concatenate([th_mem, th_osc])
        else:  # pure decay (AX0)
            r = 1.0 - np.logspace(np.log10(cfg.decay_lo),
                                  np.log10(cfg.decay_hi), H)
            theta = np.zeros(H)
        return r.astype(np.float64), theta.astype(np.float64)

    def set_schedule(self, r, theta):
        """Set per-head decay/frequency and (re)build the constant mask."""
        self.head_r = np.asarray(r, np.float64)
        self.head_theta = np.asarray(theta, np.float64)
        H, T = self.cfg.n_head, self.cfg.block_size
        t = np.arange(T)
        diff = t[:, None] - t[None, :]                  # (T,T): t - s
        causal = (diff >= 0)[None]                       # (1,T,T)
        dd = np.clip(diff, 0, None)[None]                # (1,T,T)
        D = np.where(causal,
                     (self.head_r[:, None, None] ** dd) *
                     np.cos(self.head_theta[:, None, None] * dd),
                     0.0).astype(np.float32)             # (H,T,T)
        self._D = D.reshape(1, H, T, T)

    # -- mode + params -----------------------------------------------------
    def train(self): self.training = True; return self
    def eval(self): self.training = False; return self

    def parameters(self):
        ps = [self.tok_emb, self.lnf_g, self.lnf_b]
        for b in self.blocks:
            ps.extend(b.values())
        return ps

    def num_params(self):
        return sum(p.data.size for p in self.parameters())

    def _drop(self, x):
        if not self.training or self.dropout <= 0:
            return x
        keep = 1.0 - self.dropout
        mask = (np.random.rand(*x.shape) < keep).astype(np.float32) / keep
        return x * Tensor(mask, requires_grad=False)

    def _heads(self, x, B, T):
        H, hd = self.cfg.n_head, self.head_dim
        return x.reshape(B, T, H, hd).swapaxes(1, 2)     # (B,H,T,hd)

    def _retention(self, q, k, v, T):
        """(B,H,T,hd) -> (B,H,T,hd). Chunked O(T) form when enabled and T fits;
        otherwise the parallel O(T^2) form. Both give identical results."""
        if self.chunked and T % self.chunk_size == 0:
            from chunked import chunked_retention
            return chunked_retention(q, k, v, self.head_r, self.head_theta,
                                     self.scale, self.chunk_size)
        att = (q @ k.swapaxes(-1, -2)) * self.scale
        att = att * Tensor(self._D[:, :, :T, :T], requires_grad=False)  # phasor decay+causal
        return att @ v

    def _block(self, x, blk, B, T):
        C = self.cfg.n_embd
        h = layernorm(x, blk["ln1_g"], blk["ln1_b"])
        hs = Tensor(self._shift[:T, :T], requires_grad=False) @ h
        mu = blk["mix"].sigmoid()
        hm = h * mu + hs * ((mu * -1.0) + 1.0)           # token-shift mix

        q = self._heads(hm @ blk["Wq"], B, T)
        k = self._heads(hm @ blk["Wk"], B, T)
        v = self._heads(hm @ blk["Wv"], B, T)
        o = self._retention(q, k, v, T).swapaxes(1, 2).reshape(B, T, C)
        o = layernorm(o, blk["rn_g"], blk["rn_b"])
        gate = (hm @ blk["Wg"]).swish()
        x = x + self._drop((o * gate) @ blk["Wo"])

        h2 = layernorm(x, blk["ln2_g"], blk["ln2_b"])
        ff = ((h2 @ blk["Wu"]).gelu() * (h2 @ blk["Wv2"])) @ blk["Wd"]
        return x + self._drop(ff)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = embedding(self.tok_emb, idx)
        for blk in self.blocks:
            x = self._block(x, blk, B, T)
        x = layernorm(x, self.lnf_g, self.lnf_b)
        logits = x @ self.tok_emb.swapaxes(0, 1)         # tied head
        loss = None
        if targets is not None:
            V = self.cfg.vocab_size
            loss = cross_entropy(logits.reshape(B * T, V), targets.reshape(-1))
        return logits, loss

    @np.errstate(over="ignore", invalid="ignore")
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None,
                 stop_tokens=None, rng=None):
        rng = rng or np.random
        was_training, self.training = self.training, False
        bs = self.cfg.block_size
        out = list(idx)
        try:
            for i in range(max_new_tokens):
                ctx = np.array([out[-bs:]], dtype=np.int64)
                logits, _ = self.forward(ctx)
                log = logits.data[0, -1].astype(np.float64) / max(temperature, 1e-6)
                del logits
                if top_k is not None:
                    kth = np.sort(log)[-min(top_k, len(log))]
                    log[log < kth] = -np.inf
                log -= log.max()
                p = np.exp(log); p /= p.sum()
                nxt = int(rng.choice(len(p), p=p))
                out.append(nxt)
                if stop_tokens and nxt in stop_tokens:
                    break
                if i % 16 == 15:      # forward-only graphs leak via cycles; bound it
                    gc.collect()
        finally:
            self.training = was_training
            gc.collect()
        return out

    # -- checkpoint io -----------------------------------------------------
    def state(self):
        return {f"p{i}": p.data for i, p in enumerate(self.parameters())}

    def load_state(self, arrs):
        for i, p in enumerate(self.parameters()):
            p.data = arrs[f"p{i}"].astype(np.float32)


def save_ax0(path, model, tok):
    """Self-describing checkpoint: arch + config + schedule + weights + tokenizer."""
    cfg = model.cfg
    arrs = model.state()
    arrs["arch"] = np.array(model.arch)
    arrs["mixer"] = np.array(cfg.mixer)
    arrs["cfg_int"] = np.array([cfg.vocab_size, cfg.block_size, cfg.n_layer,
                                cfg.n_head, cfg.n_embd, cfg.ff_mult, cfg.seed],
                               dtype=np.int64)
    arrs["dropout"] = np.array([cfg.dropout], dtype=np.float64)
    arrs["head_r"] = model.head_r
    arrs["head_theta"] = model.head_theta
    arrs["tok_json"] = np.array(tok.to_json())
    np.savez(path, **arrs)


def load_ax0(path):
    """Return (model, tokenizer) from a checkpoint written by save_ax0."""
    from tokenizer import AX0Tok
    d = np.load(path, allow_pickle=True)
    ci = [int(x) for x in d["cfg_int"]]
    mixer = str(d["mixer"]) if "mixer" in d.files else "decay"
    dropout = float(d["dropout"][0]) if "dropout" in d.files else 0.0
    cfg = AX0Config(vocab_size=ci[0], block_size=ci[1], n_layer=ci[2],
                    n_head=ci[3], n_embd=ci[4], ff_mult=ci[5],
                    mixer=mixer, dropout=dropout, seed=ci[6])
    model = AX0(cfg)
    if "head_r" in d.files:
        model.set_schedule(d["head_r"], d["head_theta"])
    model.load_state({k: d[k] for k in d.files
                      if k.startswith("p") and k[1:].isdigit()})
    model.eval()
    tok = AX0Tok.from_json(str(d["tok_json"]))
    return model, tok
