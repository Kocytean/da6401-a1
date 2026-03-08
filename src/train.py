"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import wandb
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, categorical
from ann.objective_functions import accuracy_score, f1_score
import os
import json

def parse_arguments():
	"""
	Parse command-line arguments.
	
	TODO: Implement argparse with the following arguments:
	- dataset: 'mnist' or 'fashion_mnist'
	- epochs: Number of training epochs
	- batch_size: Mini-batch size
	- learning_rate: Learning rate for optimizer
	- optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
	- hidden_layers: List of hidden layer sizes
	- num_neurons: Number of neurons in hidden layers
	- activation: Activation function ('relu', 'sigmoid', 'tanh')
	- loss: Loss function ('cross_entropy', 'mse')
	- weight_init: Weight initialization method
	- wandb_project: W&B project name
	- model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
	"""
	parser = argparse.ArgumentParser(description='Train a neural network')
	parser.add_argument("-d", "--dataset", default="mnist")
	parser.add_argument("-e", "--epochs", type=int, default=5)
	parser.add_argument("-b", "--batch_size", type=int, default=64)
	parser.add_argument("-l", "--loss", default="cross_entropy")
	parser.add_argument("-o", "--optimizer", default="sgd")
	parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
	parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
	parser.add_argument("-nhl", "--num_layers", type=int, default=None)
	parser.add_argument("-sz","--hidden_size","--hidden_layers", nargs="*",type=str,default=['128','64'])

	parser.add_argument("-a", "--activation", default="relu")
	parser.add_argument("-wi", "--weight_init", default="xavier")

	parser.add_argument("-wp", "--wandb_project", default="da6401")
	parser.add_argument("-lo", "--logging_options", type=str,default=None)
	return parser.parse_args()

def load_previous_best(metric_name="f1"):

	if not os.path.exists("training_metadata.json"):
		return -1

	try:
		with open("training_metadata.json", "r") as f:
			metrics = json.load(f)

		prev_metric = metrics.get("metric_name")
		prev_score = metrics.get("best_score")

		if prev_metric == metric_name and prev_score is not None:
			return prev_score
		else:
			return -1
	except Exception:
		return -1

def main():

	args = parse_arguments()
	if args.logging_options is None:
		args.logging_options = ""
	wandb.init(project=args.wandb_project, config=vars(args))

	(X_train, y_train, X_val, y_val, X_test, y_test) = load_data(args.dataset)

	args.input_size = X_train.shape[1]
	args.output_size = y_train.shape[1]
	model = NeuralNetwork(args)

	best_model_score = load_previous_best('f1')
	grad_log_neurons = None
	for epoch in range(args.epochs):
		log_dict = {"epoch": epoch}
		train_loss = model.train(
			X_train,
			y_train,
			epochs=1,
			batch_size=args.batch_size,
		)
		if '3' in args.logging_options: # support for task 3, log grad norm
			log_dict["train_loss"] = train_loss
		if '4' in args.logging_options: # support for task 4, log grad norm
			log_dict["gradw_norm"] = np.linalg.norm(model.grad_W[-1])
			log_dict["gradb_norm"] = np.linalg.norm(model.grad_b[-1])
			log_dict["activation"] = args.activation
		if '7' in args.logging_options: # support for task 7, log train accuracy
			metrics = model.evaluate(X_train, y_train, return_logits=False)
			log_dict["train_accuracy"] = metrics["accuracy"]
		if '9' in args.logging_options: # support for task 9, log grads

			dw = model.grad_W[-2]  
			grad_mag = np.mean(np.abs(dw), axis=0)

			if grad_log_neurons is None:
				nonzero = np.where(grad_mag > 1e-8)[0]
				if len(nonzero) >= 5:
					grad_log_neurons = nonzero[:5]
				else:
					# fallback: take largest gradients
					grad_log_neurons = np.argsort(-grad_mag)[:5]

			for i in grad_log_neurons:
				log_dict[f"grad_neuron_{i}"] = grad_mag[i]
			log_dict["weight_init"] = args.weight_init


		metrics = model.evaluate(X_val, y_val, return_logits=True)
		log_dict["accuracy"] = metrics["accuracy"]
		log_dict["precision"] = metrics["precision"]
		log_dict["recall"] = metrics["recall"]
		log_dict["f1"] = metrics["f1"]

		wandb.log(log_dict)


		if metrics["f1"] > best_model_score:

			best_model_score = metrics["f1"]
			np.save("best_model.npy", model.get_weights())
			with open("best_config.json", "w") as f:
				json.dump(vars(args), f, indent=4)

			metadata = {"metric_name": "f1",
					"best_score": best_model_score,
					"dataset": args.dataset}

			with open("training_metadata.json", "w") as f:
				json.dump(metadata, f, indent=4)

	if '8' in args.logging_options: # support for task 8, log confusion matrix
		test_metrics = model.evaluate(X_test, y_test, return_logits=True)
		logits = test_metrics["logits"]

		preds = np.argmax(logits, axis=1)
		labels = np.argmax(y_test, axis=1)

		num_classes = y_test.shape[1]

		cm = np.zeros((num_classes, num_classes), dtype=int)

		for t, p in zip(labels, preds):
			cm[t, p] += 1

		fig, ax = plt.subplots(figsize=(8,6))

		im = ax.imshow(cm)

		ax.set_xlabel("Predicted")
		ax.set_ylabel("True")
		ax.set_title("Confusion Matrix")

		ax.set_xticks(range(num_classes))
		ax.set_yticks(range(num_classes))

		for i in range(num_classes):
			for j in range(num_classes):
				ax.text(j, i, cm[i, j], ha="center", va="center")

		wandb.log({"confusion_matrix": wandb.Image(fig),
			"test_accuracy": test_metrics["accuracy"],
			"test_precision": test_metrics["precision"],
			"test_recall": test_metrics["recall"],
			"test_f1": test_metrics["f1"],})
	else:
		test_metrics = model.evaluate(X_test, y_test, return_logits=False)
		wandb.log(test_metrics)

	wandb.finish()
	print("Training complete!")

if __name__ == '__main__':
	main()
