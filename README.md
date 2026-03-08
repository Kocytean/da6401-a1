# da6401-a1
DA6401 Assignment 1: Multi-Layer Perceptron for Image Classification
submission by Ayush Bahuguna CS25D006

W&B report link: https://wandb.ai/cs25d006-iitm/da6401/reports/DA6401-Assignment-1--VmlldzoxNjEzMzE2NQ 

The various tasks 2.1-10 have been performed using scripts or sweeps present in the root folder or the notebooks folder. 
All WandB reports were produced using the MNIST and Fashion MNIST datasets with their train, val and test as stipulated. Keras (for the datasets) uses a PyTorch backend on my local machine, but is compatible with the autograders TensorFlow backend (the repo contains no actual PyTorch code). 
However, the best model in this repo was trained distinctly with train+test dataset as the training dataset, val for validation and the autograders dataset was used as the test dataset for the final submitted model on which it achieved a 0.804 f1 score. 
