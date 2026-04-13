import numpy as np

from simplegrad.tensor import Tensor, backward, no_grad


def test_backward_matches_example_graph_gradients():
    x = Tensor([2.0, -1.0], requires_grad=True)
    y = Tensor([3.0, 4.0], requires_grad=True)

    w = ((x + y) * x).relu()

    backward(w)

    np.testing.assert_allclose(x.grad, np.array([7.0, 0.0], dtype=np.float32))
    np.testing.assert_allclose(y.grad, np.array([2.0, 0.0], dtype=np.float32))


def test_backward_accumulates_gradient_across_multiple_uses():
    x = Tensor([2.0, -3.0], requires_grad=True)

    out = (x * x + x).sum()

    backward(out)

    np.testing.assert_allclose(x.grad, np.array([5.0, -5.0], dtype=np.float32))


def test_backward_reduces_broadcasted_gradients_to_operand_shape():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    bias = Tensor([0.5, -1.0], requires_grad=True)

    out = (x + bias).sum()

    backward(out)

    np.testing.assert_allclose(x.grad, np.ones((2, 2), dtype=np.float32))
    np.testing.assert_allclose(bias.grad, np.array([2.0, 2.0], dtype=np.float32))


def test_no_grad_prevents_reduction_from_tracking_backward_graph():
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

    with no_grad():
        out = x.sum()

    assert out.grad_fn is None
    assert out.parents == []


def test_detach_returns_tensor_without_gradient_tracking():
    x = Tensor([1.0, 2.0], requires_grad=True)

    detached = x.detach()

    assert detached.requires_grad is False
    assert detached.grad_fn is None
    assert detached.parents == []
    np.testing.assert_allclose(detached.data, x.data)
