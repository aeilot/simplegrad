import numpy as np

class Function:
    def __init__(self):
        pass

    def forward(self):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError