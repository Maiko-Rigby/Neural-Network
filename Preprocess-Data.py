import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as ts
from tensorflow import keras

# Preprocess the data
def preprocess_MIST(X_train, X_test, y_train, y_test):
    """Preprocess MNIST data for neural network"""
    
    # Flatten 28x28 images to 784-dimensional vectors
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Normalize pixel values to [0, 1]
    X_train_norm = X_train_flat.astype('float32') / 255.0
    X_test_norm = X_test_flat.astype('float32') / 255.0
    
    # Convert labels to one-hot encoding
    def to_one_hot(labels, num_classes=10):
        one_hot = np.zeros((labels.shape[0], num_classes))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot
    
    y_train_onehot = to_one_hot(y_train)
    y_test_onehot = to_one_hot(y_test)
    
    return X_train_norm, X_test_norm, y_train_onehot, y_test_onehot

X_train_proc, X_test_proc, y_train_onehot, y_test_onehot = preprocess_MIST(X_train, X_test, y_train, y_test)

print(f"\nProcessed data shapes:")
print(f"X_train_proc: {X_train_proc.shape}")
print(f"y_train_onehot: {y_train_onehot.shape}")