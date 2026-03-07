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


def main():
	"""
	Main training function.
	"""
	
	args = parse_arguments()
	if args.logging_options is None:
		args.logging_options = ""
	wandb.init(project=args.wandb_project, config=vars(args))

	(X_train, y_train, X_val, y_val, X_test, y_test) = load_data(args.dataset)

	args.input_size = X_train.shape[1]
	args.output_size = y_train.shape[1]
	model = NeuralNetwork(args)

	best_model_score = -100
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
			dw = model.grad_W[-1]
			for neuron_id in range(5):
				grad_val = np.mean(dw[neuron_id])
				log_dict["weight_init"] = args.weight_init
				log_dict[f"grad_neuron_{neuron_id}"] = grad_val


		metrics = model.evaluate(X_val, y_val)
		# log_dict["val_accuracy"] = metrics["accuracy"]
		# log_dict["val_precision"] = metrics["precision"]
		# log_dict["val_recall"] = metrics["recall"]
		log_dict["val_f1"] = metrics["f1"]

		wandb.log(log_dict)

		# print(f"Eval: Acc={metrics['accuracy']:.4f} | f1={metrics['f1']:.4f}")

		# save best model
		if metrics["f1"] > best_model_score:
			best_model_score = metrics["f1"]

			np.save("best_model.npy", model.get_weights())

			with open("best_config.json", "w") as f:
				json.dump(vars(args), f, indent=4)

	test_metrics = model.evaluate(X_test, y_test, return_logits=False)

	print("Test:", test_metrics)

	wandb.log({
		# "test_accuracy": test_metrics["accuracy"],
		# "test_precision": test_metrics["precision"],
		# "test_recall": test_metrics["recall"],
		"test_f1": test_metrics["f1"],
	})

	wandb.finish()
	print("Training complete!")

if __name__ == '__main__':
	main()
