"""
Optimization Algorithms
Implements: SGD, Momentum, Adam, Nadam, etc.
"""

import numpy as np
from abc import ABC, abstractmethod

class Optimizer(ABC):
	@abstractmethod
	def step(self, layers):
		pass

class SGD(Optimizer):
	def __init__(self, lr=0.01, decay=0.0):
		self.lr = lr
		self.decay = decay

	def step(self, layers):
		for layer in layers:
			if hasattr(layer, "W"):
				layer.W -= self.lr * (layer.dw + self.decay*layer.W)
				layer.b -= self.lr * layer.db

class Momentum(Optimizer):
	def __init__(self, lr=0.01, beta=0.9, decay=0.0):
		self.lr = lr
		self.beta = beta
		self.decay = decay

		self.vw = {}
		self.vb = {}

	def step(self, layers):
		for i, layer in enumerate(layers):
			if hasattr(layer, "W"):
				if i not in self.vw:
					self.vw[i] = np.zeros_like(layer.W)
					self.vb[i] = np.zeros_like(layer.b)

				self.vw[i] = self.beta*self.vw[i] + (layer.dw + self.decay*layer.W)
				self.vb[i] = self.beta*self.vb[i] + layer.db

				layer.W -= self.lr * self.vw[i]
				layer.b -= self.lr * self.vb[i]

class NAG(Optimizer):
	def __init__(self, lr=0.01, beta=0.9, decay=0.0):
		self.lr = lr
		self.beta = beta
		self.decay = decay

		self.vw = {}
		self.vb = {}

	def step(self, layers):
		for i, layer in enumerate(layers):
			if hasattr(layer, "W"):

				if i not in self.vw:
					self.vw[i] = np.zeros_like(layer.W)
					self.vb[i] = np.zeros_like(layer.b)

				old_vw = self.vw[i]
				old_vb = self.vb[i]

				self.vw[i] = self.beta*self.vw[i] + layer.dw + self.decay * layer.W
				self.vb[i] = self.beta*self.vb[i] + layer.db

				layer.W -= self.lr*(self.beta*old_vw + (1-self.beta)*self.vw[i])
				layer.b -= self.lr*(self.beta*old_vb + (1-self.beta)*self.vb[i])

class RMSProp(Optimizer):
	def __init__(self, lr=0.001, beta=0.9, eps=1e-8, decay=0.0):
		self.lr = lr
		self.beta = beta
		self.eps = eps
		self.decay = decay

		self.sw = {}
		self.sb = {}

	def step(self, layers):
		for i, layer in enumerate(layers):
			if hasattr(layer, "W"):

				if i not in self.sw:
					self.sw[i] = np.zeros_like(layer.W)
					self.sb[i] = np.zeros_like(layer.b)

				dw = layer.dw + self.decay*layer.W
				db = layer.db

				self.sw[i] = (self.beta*self.sw[i] + (1-self.beta)*(dw*dw))

				self.sb[i] = (self.beta*self.sb[i] + (1-self.beta)*(db*db))

				layer.W -= self.lr*dw/(np.sqrt(self.sw[i]) + self.eps)
				layer.b -= self.lr*db/(np.sqrt(self.sb[i]) + self.eps)

def optimizer(name, lr, decay=0.0):
	n = name.lower()
	if n == "sgd":
		return SGD(lr, decay)
	elif n == "momentum":
		return Momentum(lr, decay=decay)
	elif n == "nag":
		return NAG(lr, decay=decay)
	elif n == "rmsprop":
		return RMSProp(lr, decay=decay)
	raise ValueError(f"Unknown optimizer: {name}")