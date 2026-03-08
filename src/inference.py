"""
Inference Script
Evaluate trained models on test sets
"""
import os
import json	
import argparse
import numpy as np
from ann.neural_network import NeuralNetwork
from ann.objective_functions import objective_fn
from utils.data_loader import load_data, categorical
def parse_arguments():
	"""
	Parse command-line arguments for inference.
	
	TODO: Implement argparse with:
	- model_path: Path to saved model weights(do not give absolute path, rather provide relative path)
	- dataset: Dataset to evaluate on
	- batch_size: Batch size for inference
	- hidden_layers: List of hidden layer sizes
	- num_neurons: Number of neurons in hidden layers
	- activation: Activation function ('relu', 'sigmoid', 'tanh')
	"""
	parser = argparse.ArgumentParser(description='Run inference on test set')
	parser.add_argument("--model_path", default="best_model.npy")
	parser.add_argument("-d", "--dataset", default="mnist")
	parser.add_argument("-e", "--epochs", type=int, default=5)
	parser.add_argument("-b", "--batch_size", type=int, default=64)
	parser.add_argument("-l", "--loss", default="cross_entropy")
	parser.add_argument("-o", "--optimizer", default="sgd")
	parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
	parser.add_argument("-wd", "--weight_decay", type=float, default=0.0)
	parser.add_argument("-nhl", "--num_layers", type=int, default=None)
	parser.add_argument("-sz", "--hidden_size", "--hidden_layers", nargs="*", type=str)
	parser.add_argument("-a", "--activation", default="relu")
	parser.add_argument("-wi", "--weight_init", default="xavier")
	parser.add_argument("-wp", "--wandb_project", default="da6401")
	return parser.parse_args()


def load_model(model_path):
	"""
	Load trained model from disk.
	"""
	raise RuntimeError(f"DEBUG {model_path}")
	return np.load(model_path, allow_pickle=True).item()


def evaluate_model(model, X_test, y_test): 
	"""
	Evaluate model on test data.
		
	TODO: Return Dictionary - logits, loss, accuracy, f1, precision, recall
	"""
	model, loss = model
	return model.evaluate(X_test, y_test, return_logits = True, loss_fn=loss)

def fill_args_from_config(args, config_path="best_config.json"):
    """
    Fill missing CLI args from best_config.json.
    CLI arguments take precedence.
    """

    if not os.path.exists(config_path):
        return args

    with open(config_path, "r") as f:
        config = json.load(f)

    for key, value in config.items():
        if getattr(args, key, None) in [None, []]:
            setattr(args, key, value)

    return args

def main():
	"""
	Main inference function.	

	TODO: Must return Dictionary - logits, loss, accuracy, f1, precision, recall
	"""

	args = fill_args_from_config(parse_arguments())
	_, _, _, _, X_test, y_test = load_data(args.dataset)
	args.input_size = X_test.shape[1]
	args.output_size = y_test.shape[1]
	model = NeuralNetwork(args)
	weights = load_model(args.model_path)
	model.set_weights(weights)
	
	eval_results = evaluate_model((model, args.loss), X_test, y_test)
	print("Evaluation complete!")
	return eval_results

if __name__ == '__main__':
	main()
