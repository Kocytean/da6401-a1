"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np

def xavier_initializer(input_size, output_size):
	return np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size)), np.zeros((1, output_size))

def zero_initializer(input_size, output_size):
	return np.zeros((input_size, output_size)), np.zeros((1, output_size))

class Dense:

	def __init__(self, input_size, output_size, initializer=xavier_initializer):

		self.input_size = input_size
		self.output_size = output_size

		self.W, self.b = initializer(input_size, output_size)
		self.dw = np.zeros_like(self.W)
		self.db = np.zeros_like(self.b)
		self.input = None


	def forward(self, x):
		output = x @ self.W + self.b
		return output

	def backward(self, grad):

		self.dw = (self.input.T @ grad)/ batch_size
		self.db = np.sum(grad, axis=0, keepdims=True)/ batch_size
		return grad @ self.W.T

	def get_params(self):
		return {"W": self.W, "b": self.b}

	def set_params(self, params):
		self.W = params["W"]
		self.b = params["b"]

def initializer(name):
	n = name.lower()
	if n == "xavier":
		return xavier_initializer
	elif n == "zero":
		return zero_initializer 
	raise ValueError(f"Unknown initializer function: {name}")