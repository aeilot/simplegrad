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

## Example

The repository includes `example.py`:

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
python example.py
```

## API Sketch

Main entry points:

- `simplegrad.tensor.Tensor`
- `simplegrad.tensor.backward`

Useful tensor methods and operators:

- `Tensor(..., requires_grad=True)`
- `a + b`
- `a - b`
- `a * b`
- `a.relu()`
- `tensor.zero_grad()`
- `tensor.detach()`
- `with tensor.no_grad(): ...`

## Project Layout

- `simplegrad/tensor.py`: `Tensor` implementation and backpropagation
- `simplegrad/ops.py`: operation-specific backward rules
- `simplegrad/function.py`: base `Function` abstraction
- `example.py`: small runnable demo

## Status

This is a compact educational implementation, not a full deep learning framework.
