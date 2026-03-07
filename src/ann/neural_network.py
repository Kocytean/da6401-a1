import numpy as np
from ann.neural_layer import Dense, initializer
from ann.activations import activation_fn
from ann.objective_functions import objective_fn
from ann.optimizers import optimizer


class NeuralNetwork:

	def __init__(self, args):

		hidden_sizes = [int(x) for x in args.hidden_size]

		self.layers = []
		self.activations = []

		weight_init = initializer(args.weight_init)
		if hasattr(args, "input_size"):
			input_size = args.input_size
		else:
			input_size = 784
		if hasattr(args, "output_size"):
			output_size = args.output_size
		else:
			output_size = 10

		if args.num_layers is not None:
			num_layers = args.num_layers
			if hidden_sizes:
				assert len(hidden_sizes) == num_layers, "num_layers does not match hidden_sizes"
		else:
			num_layers = len(hidden_sizes)

		for h in hidden_sizes:

			self.layers.append(Dense(input_size, h, weight_init))
			self.activations.append(activation_fn(args.activation))

			input_size = h

		self.layers.append(Dense(input_size, args.output_size, weight_init))

		self.loss = objective_fn(args.loss)
		self.optimizer = optimizer(args.optimizer, args.learning_rate, args.weight_decay)

	def forward(self, X):

		for i in range(len(self.activations)):
			X = self.layers[i].forward(X)
			X = self.activations[i].forward(X)

		logits = self.layers[-1].forward(X)

		return logits

	def backward(self, y_true, y_pred):

		grad_W_list = []
		grad_b_list = []

		self.loss.forward(y_pred, y_true)
		grad = self.loss.backward()

		grad = self.layers[-1].backward(grad)

		grad_W_list.append(self.layers[-1].dw)
		grad_b_list.append(self.layers[-1].db)

		for i in range(len(self.activations)-1, -1, -1):

			grad = self.activations[i].backward(grad)
			grad = self.layers[i].backward(grad)

			grad_W_list.append(self.layers[i].dw)
			grad_b_list.append(self.layers[i].db)

		self.grad_W = np.empty(len(grad_W_list), dtype=object)
		self.grad_b = np.empty(len(grad_b_list), dtype=object)

		for i,(gw,gb) in enumerate(zip(grad_W_list, grad_b_list)):
			self.grad_W[i] = gw
			self.grad_b[i] = gb

		return self.grad_W, self.grad_b

	def update_weights(self):
		self.optimizer.step(self.layers)

	def train(self, X, y, epochs=1, batch_size=32):

		N = X.shape[0]
		running_loss = 0

		for _ in range(epochs):

			perm = np.random.permutation(N)
			X = X[perm]
			y = y[perm]

			for i in range(0, N, batch_size):

				xb = X[i:i+batch_size]
				yb = y[i:i+batch_size]

				logits = self.forward(xb)

				running_loss += self.loss.forward(logits, yb)

				self.backward(yb, logits)
				self.update_weights()

		return running_loss

	def evaluate(self, X, y, return_logits=True):

		logits = self.forward(X)

		preds = np.argmax(logits, axis=1)
		labels = np.argmax(y, axis=1)

		from ann.objective_functions import accuracy_score, f1_score

		metrics = {
			"accuracy": accuracy_score(labels, preds),
			"f1": f1_score(labels, preds)
		}

		if return_logits:
			metrics["logits"] = logits

		return metrics

	def get_weights(self):

		d = {}
		for i,layer in enumerate(self.layers):
			d[f"W{i}"] = layer.W.copy()
			d[f"b{i}"] = layer.b.copy()

		return d

	def set_weights(self, weight_dict):

		for i,layer in enumerate(self.layers):

			if f"W{i}" in weight_dict:
				layer.W = weight_dict[f"W{i}"].copy()

			if f"b{i}" in weight_dict:
				layer.b = weight_dict[f"b{i}"].copy()