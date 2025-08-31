import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as ts
from tensorflow import keras

# Preprocess the data
def preprocess_MNIST(X_train, X_test, y_train, y_test):
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

def show_preprocessing_effects(X_train, X_train_proc, y_train):
  
    # Pick a sample image
    sample_idx = 42
    original_image = X_train[sample_idx]
    normalized_image = X_train_proc[sample_idx].reshape(28, 28)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title(f'Original Image\nLabel: {y_train[sample_idx]}\nPixel range: {original_image.min()}-{original_image.max()}')
    axes[0].axis('off')
    
    # Normalised image  
    axes[1].imshow(normalized_image, cmap='gray')
    axes[1].set_title(f'Normalised Image\nLabel: {y_train[sample_idx]}\nPixel range: {normalized_image.min():.3f}-{normalized_image.max():.3f}')
    axes[1].axis('off')
    
    # Flattened representation (first 100 pixels)
    axes[2].plot(X_train_proc[sample_idx][:100])
    axes[2].set_title(f'Flattened Vector\n(First 100 of 784 values)\nShape: {X_train_proc[sample_idx].shape}')
    axes[2].set_xlabel('Pixel Index')
    axes[2].set_ylabel('Normalised Value')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
