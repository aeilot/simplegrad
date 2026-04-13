import numpy as np

class SGD:

    def __init__(self, parameters, lr=0.01, momentum=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum

        self.velocities = {id(p): np.zeros_like(p.data) for p in self.parameters}

    def step(self):
        for p in self.parameters:
            if p.grad is None:
                continue

            pid = id(p)
            grad = p.grad

            if self.momentum != 0.0:
                self.velocities[pid] = self.momentum * self.velocities[pid] + grad
                update_dir = self.velocities[pid]
            else:
                update_dir = grad

            p.data -= self.lr * update_dir

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()


class Adam:

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        self.m = {id(p): np.zeros_like(p.data) for p in self.parameters}
        self.v = {id(p): np.zeros_like(p.data) for p in self.parameters}

    def step(self):
        self.t += 1
        for p in self.parameters:
            if p.grad is None:
                continue

            pid = id(p)
            grad = p.grad

            self.m[pid] = self.beta1 * self.m[pid] + (1 - self.beta1) * grad

            self.v[pid] = self.beta2 * self.v[pid] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[pid] / (1 - self.beta1 ** self.t)
            v_hat = self.v[pid] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()