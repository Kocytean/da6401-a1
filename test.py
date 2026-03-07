import sys
import numpy as np

sys.path.append("src")

from ann.neural_network import NeuralNetwork


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


np.random.seed(0)

X = np.random.randn(5,4)
y = np.zeros((5,2))
y[np.arange(5), np.random.randint(0,2,5)] = 1

model = NeuralNetwork(Args())

logits = model.forward(X)
model.loss.forward(logits, y)
grad_W, grad_b = model.backward(y, logits)

layer = model.layers[0]
W = layer.W

eps = 1e-5
num_grad = np.zeros_like(W)

for i in range(W.shape[0]):
    for j in range(W.shape[1]):

        original = W[i,j]

        W[i,j] = original + eps
        loss_plus = model.loss.forward(model.forward(X), y)

        W[i,j] = original - eps
        loss_minus = model.loss.forward(model.forward(X), y)

        num_grad[i,j] = (loss_plus - loss_minus)/(2*eps)

        W[i,j] = original

print("Numerical grad mean:", np.mean(num_grad))
print("Backprop grad mean:", np.mean(grad_W[-1]))
print("Mean absolute error:", np.mean(np.abs(num_grad-grad_W[-1])))