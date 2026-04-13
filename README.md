# simplegrad

`simplegrad` is a small automatic differentiation project built in Python on top of NumPy.

It provides a minimal `Tensor` type with gradient tracking, a reverse-mode `backward` pass, and a few core tensor operations intended for learning and experimentation.

## Features

- NumPy-backed tensor storage
- Optional gradient tracking with `requires_grad=True`
- Reverse-mode autodiff via `backward(output)`
- Basic elementwise operations:
  - addition
  - subtraction
  - multiplication
  - ReLU
- Matrix multiplication with `@`
- Reductions with `sum()` and `mean()`
- `softmax()` and `log()` for simple loss construction
- Small neural-network helpers in `simplegrad.nn`

## Requirements

- Python `>=3.13`
- NumPy

Dependencies are declared in `pyproject.toml`.

## Install

With `uv`:

```bash
uv sync
```

Or with `pip` in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Examples

The repository includes a basic tensor/autodiff demo in `example.py`:

```python
from simplegrad.tensor import Tensor, backward

x = Tensor([2.0, -1.0], requires_grad=True)
y = Tensor([3.0, 4.0], requires_grad=True)

z = (x + y) * x
w = z.relu()

backward(w)

print(w)
print(x.grad)
print(y.grad)
```

Run it with:

```bash
uv run python example.py
```

It also includes `example-nn.py`, which trains a small two-layer neural network to learn XOR using `Linear`, `ReLU`, `SoftmaxLoss`, and `Adam`:

```bash
uv run python example-nn.py
```

The script prints the final loss, class predictions, accuracy, and output probabilities for the four XOR inputs.

## Testing

Run the test suite with:

```bash
uv run pytest
```

If you are using an activated virtual environment instead of `uv`, `python -m pytest` also works.

## API Sketch

Main entry points:

- `simplegrad.tensor.Tensor`
- `simplegrad.tensor.backward`
- `simplegrad.tensor.no_grad`

Useful tensor methods and operators:

- `Tensor(..., requires_grad=True)`
- `a + b`
- `a - b`
- `a * b`
- `a @ b`
- `a.relu()`
- `tensor.sum(...)`
- `tensor.mean(...)`
- `tensor.softmax(axis=-1)`
- `tensor.log()`
- `tensor.zero_grad()`
- `tensor.detach()`
- `with no_grad(): ...`

## Project Layout

- `simplegrad/tensor.py`: `Tensor` implementation and backpropagation
- `simplegrad/ops.py`: operation-specific backward rules
- `simplegrad/function.py`: base `Function` abstraction
- `simplegrad/nn/`: minimal modules, losses, and optimizers
- `tests/`: pytest coverage for tensor, module, loss, and optimizer behavior
- `example.py`: small runnable demo
- `example-nn.py`: XOR training example using `simplegrad.nn`

## Status

This is a compact educational implementation, not a full deep learning framework.
