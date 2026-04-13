import numpy as np
from contextlib import contextmanager
import simplegrad.ops as ops

def ensure_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)

def backward(output):
    def toposort(tensor):
        visited = set()
        order = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for pa in node.parents:
                dfs(pa)
            order.append(node)

        dfs(tensor)
        order.reverse()
        return order

    order = toposort(output)

    output.grad = np.ones_like(output.data)

    for node in order:
        if node.grad_fn is None:
            continue

        grads = node.grad_fn.backward(node.grad)

        for p, g in zip(node.parents, grads):
            if p.grad is None:
                p.grad = g
            else:
                p.grad += g

class Tensor:
    def __init__(self, data, requires_grad: bool = False):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        self.data = data
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(data)
        self.parents = []
        self.grad_fn = None
        self._computed = data is not None

    def __repr__(self):
        return f"Tensor({self.data}, grad={self.grad})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @shape.setter
    def shape(self, shape):
        self.data = self.data.reshape(shape)

    @contextmanager
    def no_grad(self):
        if self.requires_grad:
            self.requires_grad = False
            yield
            self.requires_grad = True
        else:
            yield

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        other = ensure_tensor(other)

        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad)

        if out.requires_grad:
            fn = ops.Add(self, other)
            out.grad_fn = fn
            out.parents = [self, other]

        return out

    def __radd__(self, other):
        other = ensure_tensor(other)

        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad)

        if out.requires_grad:
            fn = ops.Add(self, other)
            out.grad_fn = fn
            out.parents = [self, other]

        return out

    def __sub__(self, other):
        other = ensure_tensor(other)

        out = Tensor(self.data - other.data,
                     requires_grad=self.requires_grad or other.requires_grad)

        if out.requires_grad:
            fn = ops.Sub(self, other)
            out.grad_fn = fn
            out.parents = [self, other]

        return out

    def __rsub__(self, other):
        other = ensure_tensor(other)

        out = Tensor(other.data - self.data,
                     requires_grad=self.requires_grad or other.requires_grad)

        if out.requires_grad:
            fn = ops.Sub(other, self)
            out.grad_fn = fn
            out.parents = [other, self]

        return out

    def __mul__(self, other):
        other = ensure_tensor(other)

        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        if out.requires_grad:
            fn = ops.Mul(self, other)
            out.grad_fn = fn
            out.parents = [self, other]
        return out

    def __rmul__(self, other):
        other = ensure_tensor(other)

        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        if out.requires_grad:
            fn = ops.Mul(other, self)
            out.grad_fn = fn
            out.parents = [other, self]

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data),
                     requires_grad=self.requires_grad)
        if out.requires_grad:
            fn = ops.ReLU(self)
            out.grad_fn = fn
            out.parents = [self]

        return out

    def detach(self):
        return Tensor(self.data, requires_grad=False)