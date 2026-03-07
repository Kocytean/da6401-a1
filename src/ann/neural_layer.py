import numpy as np

def xavier_initializer(input_size, output_size):
    W = np.random.randn(input_size, output_size) * np.sqrt(1.0 / input_size)
    b = np.zeros((1, output_size))
    return W, b

def zero_initializer(input_size, output_size):
    return np.zeros((input_size, output_size)), np.zeros((1, output_size))

def initializer(name):
    n = name.lower()
    if n == "xavier":
        return xavier_initializer
    elif n == "zero":
        return zero_initializer
    raise ValueError(f"Unknown initializer: {name}")

class Dense:

    def __init__(self, input_size, output_size, initializer=xavier_initializer):

        self.W, self.b = initializer(input_size, output_size)

        self.input = None
        self.dw = None
        self.db = None

    def forward(self, x):
        self.input = x
        return x @ self.W + self.b

    def backward(self, grad):

        self.dw = self.input.T @ grad
        self.db = np.sum(grad, axis=0, keepdims=True)

        return grad @ self.W.T