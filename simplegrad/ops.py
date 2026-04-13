from simplegrad.function import Function
import numpy as np

class Add(Function):
    def __init__(self, a, b):
        super().__init__()
        self.parents = (a, b)

    def forward(self):
        return self.parents[0].data + self.parents[1].data

    def backward(self, grad_output):
        return grad_output, grad_output

class Sub(Function):
    def __init__(self, a, b):
        super().__init__()
        self.parents = (a, b)

    def forward(self):
        return self.parents[0].data - self.parents[1].data

    def backward(self, grad_output):
        return grad_output, -grad_output

class Mul(Function):
    def __init__(self, a, b):
        super().__init__()
        self.parents = (a, b)

    def forward(self):
        return self.parents[0].data * self.parents[1].data

    def backward(self, grad_output):
        x, y = self.parents
        return grad_output * y.data, grad_output * x.data


class ReLU(Function):
    def __init__(self, a):
        super().__init__()
        self.parents = (a,)

    def forward(self):
        return np.maximum(0, self.parents[0].data)

    def backward(self, grad_output):
        x = self.parents[0].data

        return grad_output * (x > 0)

class MatMul(Function):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def backward(self, grad):
        # Y = A @ B
        # dA = grad @ B.T
        # dB = A.T @ grad
        grad_a = grad @ self.b.data.T
        grad_b = self.a.data.T @ grad
        return [grad_a, grad_b]

def _get_keepdims_shape(original_shape, axis):
    if axis is None:
        return (1,) * len(original_shape)
    shape = list(original_shape)
    axes = (axis,) if isinstance(axis, int) else axis
    for a in axes:
        shape[a] = 1
    return tuple(shape)

class Sum(Function):
    def __init__(self, t, axis, keepdims):
        super().__init__()
        self.t = t
        self.axis = axis
        self.keepdims = keepdims
        self.reshaped_shape = _get_keepdims_shape(t.shape, axis)

    def backward(self, grad):
        grad = grad.reshape(self.reshaped_shape)
        return [grad * np.ones_like(self.t.data)]


class Mean(Function):
    def __init__(self, t, axis, keepdims):
        super().__init__()
        self.t = t
        self.axis = axis
        self.keepdims = keepdims
        self.reshaped_shape = _get_keepdims_shape(t.shape, axis)

        if axis is None:
            self.n_elements = t.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            self.n_elements = np.prod([t.shape[a] for a in axes])

    def backward(self, grad):
        grad = grad.reshape(self.reshaped_shape)
        return [grad * np.ones_like(self.t.data) / self.n_elements]


class Softmax(Function):
    def __init__(self, t, axis=-1):
        super().__init__()
        self.t = t
        self.axis = axis

        x_data = t.data
        shifted_x = x_data - np.max(x_data, axis=axis, keepdims=True)
        exps = np.exp(shifted_x)
        self.out_data = exps / np.sum(exps, axis=axis, keepdims=True)

    def backward(self, grad):
        S = self.out_data
        sum_grad_s = np.sum(grad * S, axis=self.axis, keepdims=True)
        grad_t = S * (grad - sum_grad_s)
        return [grad_t]

class Log(Function):
    def __init__(self, t):
        super().__init__()
        self.t = t

    def backward(self, grad):
        return [grad / (self.t.data + 1e-8)]