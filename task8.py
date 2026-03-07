import sys
from pathlib import Path

# add src folder to python path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

import json
import numpy as np
import wandb

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data


def main():

	# load saved config
	with open("best_config.json", "r") as f:
		config = json.load(f)

	wandb.init(
		project=config.get("wandb_project", "da6401"),
		name="task2_8_confusion_matrix",
		config=config
	)

	# load dataset
	X_train, y_train, X_val, y_val, X_test, y_test = load_data(config["dataset"])

	config["input_size"] = X_train.shape[1]
	config["output_size"] = y_train.shape[1]

	# convert config dict → args object
	class Args:
		pass

	args = Args()
	for k, v in config.items():
		setattr(args, k, v)

	# recreate model
	model = NeuralNetwork(args)

	# load best weights
	weights = np.load("best_model.npy", allow_pickle=True).item()
	model.set_weights(weights)

	# run inference
	metrics = model.evaluate(X_test, y_test, return_logits=True)

	logits = metrics["logits"]

	preds = np.argmax(logits, axis=1)
	labels = np.argmax(y_test, axis=1)

	num_classes = y_test.shape[1]

	class_names = [str(i) for i in range(num_classes)]

	mis_idx = np.where(preds != labels)[0][:25]

	images = X_test[mis_idx]
	true_labels = labels[mis_idx]
	pred_labels = preds[mis_idx]

	table = wandb.Table(columns=["image", "true", "pred"])

	for img, t, p in zip(images, true_labels, pred_labels):
		img = img.reshape(28,28)
		table.add_data(wandb.Image(img), int(t), int(p))

	wandb.log({"misclassified_examples": table})
	# log wandb confusion matrix
	wandb.log({
		"confusion_matrix": wandb.plot.confusion_matrix(
			y_true=labels,
			preds=preds,
			class_names=class_names
		)
	})
	wandb.finish()


if __name__ == "__main__":
	main()  