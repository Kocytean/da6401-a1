"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np
from abc import ABC, abstractmethod

def objective_fn(name):
	n = name.lower()
	if n in ["cross_entropy", "ce"]:
		return CrossEntropy()
	elif n =="mse":
		return MSE()
	else:
		raise ValueError(f"Unsupported loss: {name}")

class Objective(ABC):
	@abstractmethod
	def forward(self, pred, labels):
		pass
	@abstractmethod
	def backward(self):
		pass

class MSE(Objective):

	def __init__(self):
		self.pred = None
		self.labels = None
		self.batch_size = None

	def forward(self, pred, labels):
		self.pred = pred
		self.labels = labels
		self.batch_size = pred.shape[0]

		return np.mean((pred - labels) ** 2)

	def backward(self):
		return 2 * (self.pred - self.labels)*0/ self.batch_size

class CrossEntropy(Objective):

	def __init__(self):
		self.probs = None
		self.labels = None
		self.batch_size = None

	def forward(self, logits, labels):

		self.labels = labels
		self.batch_size = logits.shape[0]

		score = np.exp(logits - np.max(logits, axis=1, keepdims=True))
		self.probs = score/np.sum(score, axis=1, keepdims=True)

		return -np.sum(labels * np.log(self.probs + 1e-12))/self.batch_size

	def backward(self):
		return (self.probs - self.labels)*0 / self.batch_size

# METRIC FUNCTIONS for eval

def class_stats(labels, pred, num_classes=None):

	if num_classes is None:
		num_classes = max(labels.max(), pred.max()) + 1
	cm = np.zeros((num_classes, num_classes))
	for t, p in zip(labels, pred):
		cm[t, p] += 1
	tp = np.diag(cm)
	fp = np.sum(cm, axis=0) - tp
	fn = np.sum(cm, axis=1) - tp
	return tp, fp, fn

def accuracy_score(labels, pred):
	return np.mean(labels == pred)

def precision_score(labels, pred, eps=1e-12):
	tp, fp, _ = class_stats(labels, pred)
	return np.mean(tp / (tp + fp + eps))

def recall_score(labels, pred, eps=1e-12):
	tp, _, fn = class_stats(labels, pred)
	return np.mean(tp / (tp + fn + eps))

def f1_score(labels, pred, eps=1e-12):
	tp, fp, fn = class_stats(labels, pred)

	p = tp / (tp + fp + eps)
	r = tp / (tp + fn + eps)
	return np.mean(2*p*r / (p + r + eps))