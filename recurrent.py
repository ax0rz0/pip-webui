"""
recurrent.py -- O(1)-per-token streaming inference for the AX1 model.

The training-time retention mixer is the parallel form (a T x T matmul):

    A[t,s] = scale * (q_t . k_s) * r^(t-s) * cos(theta*(t-s)),   s <= t
    o_t    = sum_s A[t,s] v_s

Because r^(t-s) cos(theta*(t-s)) = Re( lambda^(t-s) ) with lambda = r * e^{i*theta},
the exact same result has a recurrent form with a fixed-size complex state per
head (head_dim x head_dim):

    S_t = lambda * S_{t-1} + outer(k_t, v_t)          # complex, shape (hd, hd)
    o_t = scale * q_t @ Re(S_t)

That is O(head_dim^2) work and memory PER TOKEN, independent of how long the
context is -- no growing KV cache. This is what makes AX1 cheap to serve and is
the basis of the future long-context ("[1m]") editions.

This module runs the whole model one token at a time in pure numpy (inference
only, no autograd), maintaining per-layer state:
  * the retention state S (complex),
  * the previous token's pre-norm value (for the token-shift mix).

It produces logits numerically identical to model.forward on the same sequence
(verified in __main__), just far cheaper for long generations.
"""

import numpy as np


# --- pointwise numpy ops matching engine.py exactly (float32) ---
def _ln(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdims=True)
    return xc * (var + eps) ** -0.5 * g + b


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _swish(x):
    return x * _sigmoid(x)


def _gelu(x):
    c = np.float32(np.sqrt(2.0 / np.pi))
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


class RecurrentAX0:
    """Streaming O(1)-state inference wrapper around a trained AX0/AX1 model."""

    def __init__(self, model):
        assert model.arch in ("ax0", "ax1"), "recurrent path is for AX0/AX1"
        self.m = model
        self.cfg = model.cfg
        self.H = model.cfg.n_head
        self.hd = model.head_dim
        self.scale = np.float32(model.scale)
        # complex per-head decay lambda = r * e^{i theta}
        self.lam = (model.head_r * np.exp(1j * model.head_theta)).astype(np.complex64)
        self.reset()

    def reset(self):
        L, H, hd, C = self.cfg.n_layer, self.H, self.hd, self.cfg.n_embd
        self.S = [np.zeros((H, hd, hd), dtype=np.complex64) for _ in range(L)]
        self.h_prev = [np.zeros(C, dtype=np.float32) for _ in range(L)]

    @np.errstate(divide="ignore", over="ignore", invalid="ignore")
    def step(self, token_id):
        """Consume one token, return next-token logits (vocab,). O(1) in context."""
        m = self.m
        H, hd = self.H, self.hd
        x = m.tok_emb.data[token_id].astype(np.float32)          # (C,)
        for li, blk in enumerate(m.blocks):
            h = _ln(x, blk["ln1_g"].data, blk["ln1_b"].data)
            mu = _sigmoid(blk["mix"].data)
            hm = mu * h + (1.0 - mu) * self.h_prev[li]           # token-shift
            self.h_prev[li] = h
            q = (hm @ blk["Wq"].data).reshape(H, hd)
            k = (hm @ blk["Wk"].data).reshape(H, hd)
            v = (hm @ blk["Wv"].data).reshape(H, hd)
            # recurrent retention state update + read
            self.S[li] = (self.lam[:, None, None] * self.S[li]
                          + k[:, :, None] * v[:, None, :])
            o = self.scale * np.einsum("hd,hde->he", q, self.S[li].real)
            o = o.reshape(self.cfg.n_embd)
            o = _ln(o, blk["rn_g"].data, blk["rn_b"].data)
            gate = _swish(hm @ blk["Wg"].data)
            x = x + (o * gate) @ blk["Wo"].data
            h2 = _ln(x, blk["ln2_g"].data, blk["ln2_b"].data)
            ff = (_gelu(h2 @ blk["Wu"].data) * (h2 @ blk["Wv2"].data)) @ blk["Wd"].data
            x = x + ff
        x = _ln(x, m.lnf_g.data, m.lnf_b.data)
        return x @ m.tok_emb.data.T                               # (vocab,)

    def generate(self, idx, max_new_tokens, temperature=0.4, top_k=None,
                 stop_tokens=None, rng=None):
        rng = rng or np.random
        self.reset()
        out = list(idx)
        logits = None
        for tid in out:                     # prime the state on the prompt
            logits = self.step(tid)
        for _ in range(max_new_tokens):
            log = logits.astype(np.float64) / max(temperature, 1e-6)
            if top_k is not None:
                kth = np.sort(log)[-min(top_k, len(log))]
                log[log < kth] = -np.inf
            log -= log.max()
            p = np.exp(log); p /= p.sum()
            nxt = int(rng.choice(len(p), p=p))
            out.append(nxt)
            if stop_tokens and nxt in stop_tokens:
                break
            logits = self.step(nxt)
        return out


if __name__ == "__main__":
    # verify the recurrent form matches the parallel forward, then time both
    import time
    from ax0 import load_ax0
    model, tok = load_ax0("ax0_pip.npz")
    model.eval()
    rec = RecurrentAX0(model)

    seq = tok.encode_prompt([("user", "hello there how are you")])[:60]
    # parallel logits at the last position
    import numpy as np
    par_logits, _ = model.forward(np.array([seq]))
    par_last = par_logits.data[0, -1]
    # recurrent logits after consuming the same sequence
    rec.reset()
    rec_last = None
    for t in seq:
        rec_last = rec.step(t)
    err = np.abs(par_last - rec_last).max() / (np.abs(par_last).max() + 1e-8)
    print(f"parallel vs recurrent max rel err: {err:.2e}  ->  {'MATCH' if err < 1e-3 else 'MISMATCH'}")

    # speed: generate 200 tokens each way
    prompt = tok.encode_prompt([("user", "hi")])
    t0 = time.time()
    rec.generate(prompt, 200, temperature=1e-6)
    t_rec = time.time() - t0
    t0 = time.time()
    model.generate(prompt, 200, temperature=1e-6)
    t_par = time.time() - t0
    print(f"generate 200 tokens: recurrent {t_rec:.2f}s | parallel(window) {t_par:.2f}s "
          f"| {t_par/t_rec:.1f}x")
