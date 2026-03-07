import os
import sys
import numpy as np

# --- ensure src is importable ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

print("Using src path:", SRC_PATH)

# now imports work
from ann.neural_network import NeuralNetwork


def gradient_check(model, X, y, epsilon=1e-5):

    y_pred = model.forward(X)
    grad_W, grad_b = model.backward(y, y_pred)

    print("\nRunning gradient check\n")

    for l, layer in enumerate(model.layers):

        W = layer.W
        numerical = np.zeros_like(W)

        for i in range(W.shape[0]):
            for j in range(W.shape[1]):

                original = W[i, j]

                W[i, j] = original + epsilon
                loss_plus = model.loss.forward(model.forward(X), y)

                W[i, j] = original - epsilon
                loss_minus = model.loss.forward(model.forward(X), y)

                W[i, j] = original

                numerical[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

        backprop = grad_W[len(model.layers) - 1 - l]

        err = np.mean(np.abs(numerical - backprop))

        print(f"Layer {l} gradient error:", err)


if __name__ == "__main__":

    np.random.seed(0)

    X = np.random.randn(5, 4)

    y = np.zeros((5, 2))
    labels = np.random.randint(0, 2, size=5)
    y[np.arange(5), labels] = 1

    class Args:
        hidden_size = ["3"]
        num_layers = 1
        activation = "relu"
        loss = "cross_entropy"
        optimizer = "sgd"
        learning_rate = 0.01
        weight_decay = 0
        weight_init = "xavier"
        input_size = 4
        output_size = 2

    model = NeuralNetwork(Args())

    gradient_check(model, X, y)