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

class ReLU(Activation):
	def __init__(self):
		self.input = None

	def forward(self, x):
		self.input = x
		return np.maximum(0, x)

	def backward(self, grad):
		return grad * (self.input > 0)

class Sigmoid(Activation):
	def __init__(self):
		self.output = None 

	def forward(self, x):
		self.output = 1 / (1 + np.exp(-x))
		return self.output

	def backward(self, grad):
		return grad*self.output* (1 - self.output)

class Tanh(Activation):
	def __init__(self):
		self.output = None
	def forward(self, x):
		self.output = np.tanh(x)
		return self.output
	def backward(self, grad):
		return grad * (1 - self.output*self.output)

def activation_fn(name: str):
	n = name.lower()

	if n == "relu":
		return ReLU()
	elif n == "sigmoid":
		return Sigmoid()
	elif n == "tanh":
		return Tanh()
	raise ValueError(f"Unknown activation function: {name}")