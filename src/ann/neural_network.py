"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
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
				parts = item.split(",")
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
			input_size = 728
		output_size = cli_args.output_size

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
		"""
		Backward propagation to compute gradients.
		Returns two numpy arrays: grad_Ws, grad_bs.
		- `grad_Ws[0]` is gradient for the last (output) layer weights,
		  `grad_bs[0]` is gradient for the last layer biases, and so on.
		"""
		grad_W_list = []
		grad_b_list = []

		# Backprop through layers in reverse; collect grads so that index 0 = last layer
		self.loss.forward(y_pred, y_true)
		dL = self.loss.backward()
		dL = self.layers[-1].backward(dL)
		grad_W_list.append(self.layers[-1].dw)
		grad_b_list.append(self.layers[-1].db)
		for i, layer in enumerate(self.layers[:-1][::-1]):
			dL = layer.backward(self.activation_fns[-(i+1)].backward(dL))
			grad_W_list.append(layer.dw)
			grad_b_list.append(layer.db)
		# create explicit object arrays to avoid numpy trying to broadcast shapes

		self.grad_W = np.empty(len(grad_W_list), dtype=object)
		self.grad_b = np.empty(len(grad_b_list), dtype=object)
		for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
			self.grad_W[i] = gw
			self.grad_b[i] = gb

		# print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
		# print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
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

		acc = accuracy_score(labels, preds)
		precision = precision_score(labels, preds)
		recall = recall_score(labels, preds)
		f1 = f1_score(labels, preds)

		metrics =  {
			"accuracy": acc,
			"precision": precision,
			"recall": recall,
			"f1": f1}

		if loss_fn is not None:
			if isinstance(list):
				for loss in loss_fn:
					metrics[loss]=objective_fn(loss_fn).forward(logits, y)
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

