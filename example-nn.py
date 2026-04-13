import numpy as np

from simplegrad.nn.module import Module
from simplegrad.nn.nn import Linear, SoftmaxLoss
from simplegrad.nn.optim import Adam
from simplegrad.tensor import Tensor, backward


class XORNet(Module):
    def __init__(self, hidden_size=8):
        self.hidden = Linear(2, hidden_size)
        self.output = Linear(hidden_size, 2)

    def forward(self, x):
        return self.output(self.hidden(x).relu())


def train_xor(epochs=1500, lr=0.1, seed=0, hidden_size=8):
    np.random.seed(seed)

    inputs = Tensor(
        np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    labels = np.array([0, 1, 1, 0])
    targets = Tensor(np.eye(2, dtype=np.float32)[labels])

    model = XORNet(hidden_size=hidden_size)
    loss_fn = SoftmaxLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    losses = []
    for _ in range(epochs):
        logits = model(inputs)
        loss = loss_fn(logits, targets)

        optimizer.zero_grad()
        backward(loss)
        optimizer.step()

        losses.append(float(loss.data))

    logits = model(inputs)
    probabilities = logits.softmax(axis=-1)
    predictions = np.argmax(probabilities.data, axis=-1)
    accuracy = float(np.mean(predictions == labels))

    return {
        "model": model,
        "losses": losses,
        "predictions": predictions,
        "accuracy": accuracy,
        "probabilities": probabilities.data,
    }


def main():
    result = train_xor()

    print(f"final loss: {result['losses'][-1]:.4f}")
    print(f"predictions: {result['predictions']}")
    print(f"accuracy: {result['accuracy']:.2f}")
    print("probabilities:")
    print(result["probabilities"])


if __name__ == "__main__":
    main()
