"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
	@abstractmethod
	def forward(self, x):
		pass
	@abstractmethod
	def backward(self, grad):
		pass

import numpy as np

class ReLU(Activation):

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * self.mask


class Sigmoid(Activation):

    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1 - self.out)


class Tanh(Activation):

    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1 - self.out**2)


def activation_fn(name):

    n = name.lower()

    if n == "relu":
        return ReLU()
    elif n == "sigmoid":
        return Sigmoid()
    elif n == "tanh":
        return Tanh()

    raise ValueError(f"Unknown activation {name}")