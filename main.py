from NeuralNetwork import NeuralNetwork
from DatasetVisualisation import visualise_mnist_dataset
from PreprocessData import preprocess_MNIST, show_preprocessing_effects
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as ts
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data() # get the MNIST dataset

print(f"Original data shapes:")
print(f"X_train: {X_train.shape}") 
print(f"y_train: {y_train.shape}") 
print(f"X_test: {X_test.shape}")    
print(f"y_test: {y_test.shape}")

visualise_mnist_dataset(X_train, y_train)

X_train_proc, X_test_proc, y_train_onehot, y_test_onehot = preprocess_MNIST(X_train, X_test, y_train, y_test)

print(f"\nProcessed data shapes:")
print(f"X_train_proc: {X_train_proc.shape}")
print(f"y_train_onehot: {y_train_onehot.shape}") 

show_preprocessing_effects(X_train, X_train_proc, y_train_onehot)

nn_mnist = NeuralNetwork(input_size=784, hidden_size=128, output_size=10, learning_rate=0.5)
nn_mnist.debug_shapes = True

print(f"Network architecture:")
print(f"Input layer: 784 neurons (28×28 pixel values)")
print(f"Hidden layer: 128 neurons (sigmoid activation)")
print(f"Output layer: 10 neurons (softmax activation for digits 0-9)")
print(f"Total parameters: {784*128 + 128 + 128*10 + 10:,}")

losses, accuracies = nn_mnist.train(X_train_proc, y_train_onehot, epochs=1000, print_every=10)

