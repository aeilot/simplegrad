import numpy as np
from contextlib import contextmanager
import simplegrad.ops as ops

class GradMode:
    enabled = True

@contextmanager
def no_grad():
    prev = GradMode.enabled
    GradMode.enabled = False
    try:
        yield
    finally:
        GradMode.enabled = prev


def ensure_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)

def unbroadcast(grad, target_shape):
    if grad.shape == target_shape:
        return grad
    ndims_added = grad.ndim - len(target_shape)
    for _ in range(ndims_added):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

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
                p.grad += unbroadcast(g, p.shape)

class Tensor:
    def __init__(self, data, requires_grad: bool = False):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        self.data = data
        self.requires_grad = requires_grad
        self.grad = None
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

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def __add__(self, other):
        other = ensure_tensor(other)

        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad)

        if out.requires_grad and GradMode.enabled:
            fn = ops.Add(self, other)
            out.grad_fn = fn
            out.parents = [self, other]

        return out

    def __radd__(self, other):
        other = ensure_tensor(other)

        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad)

        if out.requires_grad and GradMode.enabled:
            fn = ops.Add(self, other)
            out.grad_fn = fn
            out.parents = [self, other]

        return out

    def __sub__(self, other):
        other = ensure_tensor(other)

        out = Tensor(self.data - other.data,
                     requires_grad=self.requires_grad or other.requires_grad)

        if out.requires_grad and GradMode.enabled:
            fn = ops.Sub(self, other)
            out.grad_fn = fn
            out.parents = [self, other]

        return out

    def __rsub__(self, other):
        other = ensure_tensor(other)

        out = Tensor(other.data - self.data,
                     requires_grad=self.requires_grad or other.requires_grad)

        if out.requires_grad and GradMode.enabled:
            fn = ops.Sub(other, self)
            out.grad_fn = fn
            out.parents = [other, self]

        return out

    def __mul__(self, other):
        other = ensure_tensor(other)

        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        if out.requires_grad and GradMode.enabled:
            fn = ops.Mul(self, other)
            out.grad_fn = fn
            out.parents = [self, other]
        return out

    def __rmul__(self, other):
        other = ensure_tensor(other)

        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        if out.requires_grad and GradMode.enabled:
            fn = ops.Mul(other, self)
            out.grad_fn = fn
            out.parents = [other, self]

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data),
                     requires_grad=self.requires_grad)
        if out.requires_grad and GradMode.enabled:
            fn = ops.ReLU(self)
            out.grad_fn = fn
            out.parents = [self]

        return out

    def __matmul__(self, other):
        other = ensure_tensor(other)
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        if out.requires_grad and GradMode.enabled:
            out.grad_fn = ops.MatMul(self, other)
            out.parents = [self, other]
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad)
        if out.requires_grad:
            out.grad_fn = ops.Sum(self, axis, keepdims)
            out.parents = [self]
        return out

    def mean(self, axis=None, keepdims=False):
        out = Tensor(np.mean(self.data, axis=axis, keepdims=keepdims),
                     requires_grad=self.requires_grad)
        if out.requires_grad:
            out.grad_fn = ops.Mean(self, axis, keepdims)
            out.parents = [self]
        return out

    def softmax(self, axis=-1):
        fn = ops.Softmax(self, axis)
        out = Tensor(fn.out_data, requires_grad=self.requires_grad)
        if out.requires_grad:
            out.grad_fn = fn
            out.parents = [self]
        return out

    def log(self):
        out_data = np.log(self.data + 1e-8)
        out = Tensor(out_data, requires_grad=self.requires_grad)
        if out.requires_grad:
            out.grad_fn = ops.Log(self)
            out.parents = [self]
        return out

    def __neg__(self):
        return self * -1.0

    def detach(self):
        return Tensor(self.data, requires_grad=False)