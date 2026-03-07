import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import wandb
from types import SimpleNamespace

from utils.data_loader import load_data
from ann.neural_network import NeuralNetwork
import matplotlib.pyplot as plt

def task_5(activation):

	wandb.init(
		project="da6401",
		name=f"dead-neuron-{activation}",
		config={"activation": activation}
	)

	X_train, y_train, X_val, y_val, _, _ = load_data("fashion_mnist")

	args = SimpleNamespace(input_size=X_train.shape[1],
		output_size=y_train.shape[1],
		hidden_size=["128","128","128"],
		num_layers=3,
		activation=activation,
		optimizer="rmsprop",
		learning_rate=0.1,
		weight_decay=0.0,
		loss="cross_entropy",
		weight_init="xavier")
	model = NeuralNetwork(args)

	epochs = 10
	for epoch in range(epochs):

		model.train(X_train, y_train, epochs=1, batch_size=32)
		metrics = model.evaluate(X_val, y_val)
		wandb.log({
			"epoch": epoch,
			"val_accuracy": metrics["accuracy"]
		})

	logits, activations = model.forward_trace(X_val)

	table = wandb.Table(columns=["layer", "activation_histogram"])

	for layer_id, act in enumerate(activations):
		dead_neurons = np.sum(np.all(act == 0, axis=0))
		wandb.log({f"dead_neurons_layer_{layer_id}": dead_neurons})
		fig, ax = plt.subplots()

		ax.hist(act.flatten(), bins=10, range = (-1,1) if activation=='tanh' else (0,act.flatten().max()))

		ax.set_title(f"Layer {layer_id}")
		ax.set_xlabel("Activation")
		ax.set_ylabel("Frequency")

		table.add_data(layer_id, wandb.Image(fig))

		plt.close(fig)

	wandb.log({"activation_distributions": table})

	wandb.finish()


if __name__ == "__main__":

	task_5("relu")
	task_5("tanh")