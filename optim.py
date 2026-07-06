"""
optim.py -- from-scratch optimizers for the AX model family.

Plain numpy, operating on engine.Tensor parameters (it only touches `.data` and
`.grad`). Currently just AdamW with decoupled weight decay; kept in its own
module so the AX0/AX1 stack has no dependency on the legacy GPT baseline.
"""

import numpy as np


class AdamW:
    """Adam with decoupled weight decay (Loshchilov & Hutter)."""

    def __init__(self, params, lr=3e-3, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self, lr=None):
        if lr is not None:
            self.lr = lr
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            update = mhat / (np.sqrt(vhat) + self.eps)
            # decoupled weight decay (skip 1-D params: biases / norm gains)
            if self.wd > 0 and p.data.ndim >= 2:
                update = update + self.wd * p.data
            p.data = p.data - self.lr * update
