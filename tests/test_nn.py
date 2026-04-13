import numpy as np

from simplegrad.nn.module import Module, Parameter
from simplegrad.nn.nn import Linear, MSELoss, SoftmaxLoss
from simplegrad.nn.optim import Adam, SGD
from simplegrad.tensor import Tensor, backward


class ToyModel(Module):
    def __init__(self):
        self.layer = Linear(2, 3)
        self.scale = Parameter([1.0, 2.0, 3.0])

    def forward(self, x):
        return self.layer(x) * self.scale


def test_module_parameters_recurses_into_submodules():
    model = ToyModel()

    params = model.parameters()

    assert params == [model.layer.weight, model.layer.bias, model.scale]


def test_mse_loss_backward_matches_expected_mean_square_gradient():
    predictions = Tensor([3.0, -1.0], requires_grad=True)
    targets = Tensor([1.0, 1.0])

    loss = MSELoss()(predictions, targets)

    backward(loss)

    np.testing.assert_allclose(loss.data, np.array(4.0, dtype=np.float32))
    np.testing.assert_allclose(
        predictions.grad, np.array([2.0, -2.0], dtype=np.float32)
    )


def test_softmax_loss_backward_matches_probabilities_minus_targets():
    logits = Tensor([[2.0, 0.0]], requires_grad=True)
    targets = Tensor([[1.0, 0.0]])

    loss = SoftmaxLoss()(logits, targets)

    backward(loss)

    probs = np.exp([2.0, 0.0]) / np.exp([2.0, 0.0]).sum()
    expected_grad = probs - np.array([1.0, 0.0])
    np.testing.assert_allclose(logits.grad, expected_grad.reshape(1, 2), rtol=1e-6)


def test_sgd_step_updates_parameter_data_from_gradient():
    parameter = Parameter([1.0, -2.0])
    parameter.grad = np.array([0.5, -1.0], dtype=np.float32)

    optimizer = SGD([parameter], lr=0.1)
    optimizer.step()

    np.testing.assert_allclose(parameter.data, np.array([0.95, -1.9], dtype=np.float32))


def test_adam_step_updates_parameter_data():
    parameter = Parameter([1.0, -2.0])
    parameter.grad = np.array([0.5, -1.0], dtype=np.float32)

    optimizer = Adam([parameter], lr=0.1)
    optimizer.step()

    np.testing.assert_allclose(
        parameter.data, np.array([0.9, -1.9], dtype=np.float32), rtol=1e-6
    )
