from simplegrad.nn.module import Module, Parameter
from simplegrad.tensor import ensure_tensor
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        stdv = 1.0 / np.sqrt(in_features)
        weight_data = np.random.uniform(-stdv, stdv, (in_features, out_features))
        bias_data = np.zeros(out_features)

        self.weight = Parameter(weight_data)
        self.bias = Parameter(bias_data)

    def forward(self, x):
        return x @ self.weight + self.bias


class MSELoss(Module):

    def forward(self, predictions, targets):
        targets = ensure_tensor(targets)
        diff = predictions - targets
        sq_diff = diff * diff
        return sq_diff.mean()


class SoftmaxLoss(Module):

    def forward(self, logits, targets):
        targets = ensure_tensor(targets)

        probs = logits.softmax(axis=-1)

        log_probs = probs.log()

        loss = -(targets * log_probs).sum(axis=-1).mean()

        return loss