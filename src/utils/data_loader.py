"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import os
os.environ["KERAS_BACKEND"] = "torch"
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split
import numpy as np

def categorical(y, num_classes=10):
	return np.eye(num_classes)[y]

def load_data(name):
	n = name.lower()
	if n == "mnist":
		(X_train, y_train), (X_test, y_test) = mnist.load_data()
	elif n == "fashion_mnist":
		(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
	else:
		raise ValueError(f"Unknown dataset {name}")

	X_train = X_train.reshape(len(X_train), -1) / 255.0
	X_test = X_test.reshape(len(X_test), -1) / 255.0

	X_train, X_val, y_train, y_val = train_test_split(
		X_train, y_train, test_size=0.1, random_state=42
	)

	return (X_train, categorical(y_train), X_val, categorical(y_val), X_test, categorical(y_test))