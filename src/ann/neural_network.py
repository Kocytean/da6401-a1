"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))
import numpy as np
from ann.neural_layer import *
from ann.activations import *
from ann.objective_functions import *
from ann.optimizers import *

class NeuralNetwork:

	def __init__(self, cli_args):
		def parse_hidden_sizes(arg):
			if arg is None or len(arg) == 0:
				return []
			
			sizes = []
			for item in arg:
				# split comma-separated values
				parts = str(item).split(",")
				for p in parts:
					p = p.strip()
					if p == "":
						continue
					if not p.isdigit():
						raise ValueError(f"Invalid hidden size '{p}'. Must be positive integers.")
					val = int(p)
					if val <= 0:
						raise ValueError("Hidden layer sizes must be positive.")
					sizes.append(val)
			return sizes
		self.activation_fns = []
		self.layers = []
		if hasattr(cli_args, "input_size"):
			input_size = cli_args.input_size
		else:
			input_size = 784
		if hasattr(cli_args, "output_size"):
			output_size = cli_args.output_size
		else:
			output_size = 10

		hidden_sizes = parse_hidden_sizes(cli_args.hidden_size)
		if cli_args.num_layers is not None:
			num_layers = cli_args.num_layers
			if hidden_sizes:
				assert len(hidden_sizes) == num_layers, "num_layers does not match hidden_sizes"
		else:
			num_layers = len(hidden_sizes)
		weight_init = initializer(cli_args.weight_init)
		activation_name = cli_args.activation
		
		for h in hidden_sizes:
			new_layer = Dense(input_size, h, weight_init)
			self.layers.append(new_layer)
			self.activation_fns.append(activation_fn(activation_name))
			input_size = h
		self.layers.append(Dense(input_size, output_size, weight_init))
		
		self.loss = objective_fn(cli_args.loss)
		self.optimizer = optimizer(cli_args.optimizer, cli_args.learning_rate, cli_args.weight_decay)
	
	def forward(self, X):
		for i, layer in enumerate(self.layers[:-1]):
			X = self.activation_fns[i].forward(layer.forward(X))
		return self.layers[-1].forward(X)

	def forward_trace(self, X):
		activations = []
		for i, layer in enumerate(self.layers[:-1]):
			X = self.activation_fns[i].forward(layer.forward(X))
			activations.append(X)
		return self.layers[-1].forward(X), activations

	def backward(self, y_true, y_pred):

		grad_W_list = []
		grad_b_list = []

		dL = self.loss.backward()

		# output layer
		dL = self.layers[-1].backward(dL)
		grad_W_list.append(self.layers[-1].dw)
		grad_b_list.append(self.layers[-1].db)

		# hidden layers
		for layer, activation in zip(
			reversed(self.layers[:-1]),
			reversed(self.activation_fns)
		):
			dL = activation.backward(dL)
			dL = layer.backward(dL)
			grad_W_list.append(layer.dw)
			grad_b_list.append(layer.db)

		self.grad_W = np.empty(len(grad_W_list), dtype=object)
		self.grad_b = np.empty(len(grad_b_list), dtype=object)

		for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
			self.grad_W[i] = gw
			self.grad_b[i] = gb

		return self.grad_W, self.grad_b

	def update_weights(self):
		self.optimizer.step(self.layers)

	def train(self, X_train, y_train, epochs=1, batch_size=32):
		size = X_train.shape[0]

		for epoch in range(epochs):

			shuffled_indices = np.random.permutation(size)
			X_train = X_train[shuffled_indices]
			y_train = y_train[shuffled_indices]

			running_loss = 0

			for i in range(0, size, batch_size):

				X_batch = X_train[i:i+batch_size]
				y_batch = y_train[i:i+batch_size]

				logits = self.forward(X_batch)
				batch_loss = self.loss.forward(logits, y_batch)
				running_loss += batch_loss

				self.backward(y_batch, logits)
				self.update_weights()

			print(f"{epoch+1}/{epochs} - {running_loss}")
		return running_loss

	def evaluate(self, X, y, return_logits = True, loss_fn = None):
		logits = self.forward(X)

		preds = np.argmax(logits, axis=1)
		labels = np.argmax(y, axis=1)
		# acc = accuracy_	score(labels, preds)
		# precision = precision_score(labels, preds)
		# recall = recall_score(labels, preds)
		f1 = f1_score(labels, preds)

		metrics =  {
			# "accuracy": acc,
			# "precision": precision,
			# "recall": recall,
			"f1": f1}

		if loss_fn is not None:
			if isinstance(loss_fn, list):
				for loss in loss_fn:
					metrics[loss]=objective_fn(loss).forward(logits, y)
			else:
				metrics["loss"]=objective_fn(loss_fn).forward(logits, y)
		if return_logits:
			metrics["logits"]= logits
		return metrics

	def get_weights(self):
		d = {}
		for i, layer in enumerate(self.layers):
			d[f"W{i}"] = layer.W.copy()
			d[f"b{i}"] = layer.b.copy()
		return d

	def set_weights(self, weight_dict):
		for i, layer in enumerate(self.layers):
			w_key = f"W{i}"
			b_key = f"b{i}"
			if w_key in weight_dict:
				layer.W = weight_dict[w_key].copy()
			if b_key in weight_dict:
				layer.b = weight_dict[b_key].copy()

def gradient_check(model, X, y, epsilon=1e-5):

	# Forward pass
	y_pred = model.forward(X)

	# Backprop gradients
	grad_W, grad_b = model.backward(y, y_pred)

	print("Checking gradients...\n")

	for l, layer in enumerate(model.layers):

		W = layer.W
		num_grad_W = np.zeros_like(W)

		for i in range(W.shape[0]):
			for j in range(W.shape[1]):

				original = W[i, j]

				# W + epsilon
				W[i, j] = original + epsilon
				loss_plus = model.loss.forward(model.forward(X), y)

				# W - epsilon
				W[i, j] = original - epsilon
				loss_minus = model.loss.forward(model.forward(X), y)

				# restore weight
				W[i, j] = original

				num_grad_W[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

		backprop_grad = grad_W[len(model.layers)-1-l]

		diff = np.mean(np.abs(num_grad_W - backprop_grad))

		print(f"Layer {l} weight gradient error:", diff)

if __name__ == '__main__':
	np.random.seed(0)

	X = np.random.randn(5, 4)
	y = np.zeros((5, 2))
	y[np.arange(5), np.random.randint(0,2,5)] = 1

	model = NeuralNetwork(
		layers=[Dense(4,3), Dense(3,2)],
		activation_fns=[ReLU()],
		loss=CrossEntropy()
	)

	gradient_check(model, X, y)