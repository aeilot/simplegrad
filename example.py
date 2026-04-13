from simplegrad.tensor import Tensor, backward
import numpy as np

def main():
    x = Tensor([2.0, -1.0], requires_grad=True)
    y = Tensor([3.0, 4.0], requires_grad=True)
    print(f"x: {x}")
    print(f"y: {y}\n")

    z = (x + y) * x
    print(f"z = (x + y) * x:\n{z}")

    w = z.relu()
    print(f"w = z.relu():\n{w}\n")

    backward(w)

    print(f"Gradient of x (x.grad):\n{x.grad}")
    print(f"Gradient of y (y.grad):\n{y.grad}")

if __name__ == "__main__":
    main()