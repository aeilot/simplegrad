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

class MatMul:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def backward(self, grad):
        # Y = A @ B
        # dA = grad @ B.T
        # dB = A.T @ grad
        grad_a = grad @ self.b.data.T
        grad_b = self.a.data.T @ grad
        return [grad_a, grad_b]