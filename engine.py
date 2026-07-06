"""
engine.py -- a tiny reverse-mode autodiff engine over numpy arrays.

This is the "deep learning framework" for our homemade LLM, written from
scratch. A Tensor wraps a numpy array and records the operations performed on
it so that gradients can be computed with a backward pass (the same idea behind
PyTorch's autograd, just much smaller).

Nothing here knows anything about transformers -- it only knows how to add,
multiply, matmul, and a handful of other primitives, and how to backprop
through them. ax0.py builds the actual model on top of these primitives (and
gpt.py the legacy comparison baseline).
"""

import numpy as np

# Global compute dtype. float32 is the default (fast on Accelerate BLAS);
# the gradient-check harness flips this to float64 to verify correctness.
DTYPE = np.float32


def _noop():
    pass


def _unbroadcast(grad, shape):
    """Sum `grad` down to `shape`, reversing numpy broadcasting."""
    # 1) remove leading dims that were added by broadcasting
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2) sum over dims that were size-1 in the original and got broadcast up
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    __slots__ = ("data", "grad", "_backward", "_prev", "requires_grad")

    def __init__(self, data, _children=(), requires_grad=True):
        self.data = np.asarray(data, dtype=DTYPE)
        self.grad = None
        self._backward = lambda: None
        self._prev = _children
        self.requires_grad = requires_grad

    # -- helpers -----------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    def _accumulate(self, g):
        if not self.requires_grad:
            return
        if self.grad is None:
            # avoid an unconditional copy: .astype() copies even when the dtype
            # already matches, which it almost always does here.
            self.grad = g if g.dtype == DTYPE else g.astype(DTYPE)
        else:
            self.grad = self.grad + g

    @staticmethod
    def _as_tensor(x):
        return x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)

    # -- elementwise -------------------------------------------------------
    def __add__(self, other):
        other = self._as_tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward():
            self._accumulate(_unbroadcast(out.grad, self.shape))
            other._accumulate(_unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._as_tensor(other)
        out = Tensor(self.data * other.data, (self, other))

        def _backward():
            self._accumulate(_unbroadcast(out.grad * other.data, self.shape))
            other._accumulate(_unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward
        return out

    def __pow__(self, p):
        assert isinstance(p, (int, float))
        out = Tensor(self.data ** p, (self,))

        def _backward():
            self._accumulate(_unbroadcast(out.grad * (p * self.data ** (p - 1)), self.shape))

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._as_tensor(other))

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = self._as_tensor(other)
        return self * (other ** -1.0)

    # -- matmul ------------------------------------------------------------
    def __matmul__(self, other):
        # NOTE: numpy 2.x on Apple Accelerate raises spurious "overflow/divide
        # by zero" FP flags for batched float32 matmul even when inputs and
        # outputs are perfectly finite (verified: result matches float64 to
        # ~1e-9). We silence those flags here only; a real non-finite value
        # would still be caught by the finite-loss guard in the training loop.
        other = self._as_tensor(other)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            out = Tensor(self.data @ other.data, (self, other))

        def _backward():
            a, b, g = self.data, other.data, out.grad
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                da = g @ np.swapaxes(b, -1, -2)
                db = np.swapaxes(a, -1, -2) @ g
            self._accumulate(_unbroadcast(da, self.shape))
            other._accumulate(_unbroadcast(db, other.shape))

        out._backward = _backward
        return out

    # -- shape ops ---------------------------------------------------------
    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,))

        def _backward():
            self._accumulate(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def swapaxes(self, a, b):
        out = Tensor(np.swapaxes(self.data, a, b), (self,))

        def _backward():
            self._accumulate(np.swapaxes(out.grad, a, b))

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def _backward():
            g = out.grad
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accumulate(np.broadcast_to(g, self.shape).copy())

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            n = self.data.size
        else:
            n = self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    # -- nonlinearities ----------------------------------------------------
    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), (self,))

        def _backward():
            self._accumulate(out.grad * (self.data > 0))

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,))

        def _backward():
            self._accumulate(out.grad * (1 - t * t))

        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + np.exp(-np.clip(self.data, -30.0, 30.0)))
        out = Tensor(s, (self,))

        def _backward():
            self._accumulate(out.grad * s * (1.0 - s))

        out._backward = _backward
        return out

    def swish(self):
        # x * sigmoid(x) (a.k.a. SiLU)
        sg = 1.0 / (1.0 + np.exp(-np.clip(self.data, -30.0, 30.0)))
        out = Tensor(self.data * sg, (self,))

        def _backward():
            self._accumulate(out.grad * (sg + self.data * sg * (1.0 - sg)))

        out._backward = _backward
        return out

    def gelu(self):
        # tanh approximation of GELU. Keep the constant in the compute dtype so a
        # float64 np.sqrt scalar doesn't upcast the array (NEP-50) and force a
        # float32 copy back. (dtype hygiene; not a measured hotspot.)
        c = DTYPE(np.sqrt(2.0 / np.pi))
        x = self.data
        inner = c * (x + 0.044715 * x ** 3)
        t = np.tanh(inner)
        out_val = 0.5 * x * (1.0 + t)
        out = Tensor(out_val, (self,))

        def _backward():
            sech2 = 1 - t * t
            dinner = c * (1.0 + 3 * 0.044715 * x ** 2)
            d = 0.5 * (1.0 + t) + 0.5 * x * sech2 * dinner
            self._accumulate(out.grad * d)

        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        z = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(p, (self,))

        def _backward():
            g = out.grad
            dot = (g * p).sum(axis=axis, keepdims=True)
            self._accumulate(p * (g - dot))

        out._backward = _backward
        return out

    # -- backward driver ---------------------------------------------------
    def backward(self):
        topo, visited = [], set()

        def build(v):
            if id(v) in visited:
                return
            visited.add(id(v))
            for child in v._prev:
                build(child)
            topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
        # Free the graph immediately: the _backward closures capture their own
        # output tensor, forming reference cycles that refcounting can't reclaim
        # (it would wait for cyclic GC, ballooning memory ~GB/step on big models).
        # Clearing them breaks the cycles so the graph is freed as soon as the
        # loss tensor is dropped.
        for node in topo:
            node._backward = _noop
            node._prev = ()


# -- functions that need access to integer index arrays (not differentiable
#    wrt the indices) -------------------------------------------------------

def embedding(table, idx):
    """table: Tensor (vocab, dim); idx: int ndarray (...,). -> Tensor (..., dim)."""
    out = Tensor(table.data[idx], (table,))

    def _backward():
        if not table.requires_grad:
            return
        grad = np.zeros_like(table.data)
        np.add.at(grad, idx.reshape(-1), out.grad.reshape(-1, table.shape[1]))
        table._accumulate(grad)

    out._backward = _backward
    return out


def select(t, axis, index):
    """Differentiable single-index select along `axis` (drops that axis)."""
    out = Tensor(np.take(t.data, index, axis=axis), (t,))

    def _backward():
        g = np.zeros_like(t.data)
        idx = [slice(None)] * t.data.ndim
        idx[axis] = index
        g[tuple(idx)] = out.grad
        t._accumulate(g)

    out._backward = _backward
    return out


def stack(tensors, axis=0):
    """Differentiable stack of a list of Tensors along a new `axis`."""
    out = Tensor(np.stack([t.data for t in tensors], axis=axis), tuple(tensors))

    def _backward():
        gs = np.split(out.grad, len(tensors), axis=axis)
        for t, g in zip(tensors, gs):
            t._accumulate(np.squeeze(g, axis=axis))

    out._backward = _backward
    return out


def layernorm(x, gamma, beta, eps=1e-5):
    """Layer norm over the last axis. x:(...,D) gamma,beta:(D,)."""
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    inv = (var + eps) ** -0.5
    return xc * inv * gamma + beta


def cross_entropy(logits, targets):
    """logits: Tensor (N, V). targets: int ndarray (N,). -> scalar Tensor."""
    z = logits.data - logits.data.max(axis=-1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    n = logits.shape[0]
    logp = np.log(p[np.arange(n), targets] + 1e-9)
    loss = -logp.mean()
    out = Tensor(loss, (logits,))

    def _backward():
        d = p.copy()
        d[np.arange(n), targets] -= 1.0
        d /= n
        logits._accumulate(d * out.grad)

    out._backward = _backward
    return out
