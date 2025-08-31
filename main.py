from NeuralNetwork import NeuralNetwork
from DatasetVisualisation import visualise_mnist_dataset
from PreprocessData import preprocess_MNIST, show_preprocessing_effects
import numpy as np
import matplotlib.pyplot as plt
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

# ------------------------------------------------------------------------------------------------------------------------------------------------

nn_mnist = NeuralNetwork(input_size=784, hidden_size=128, output_size=10, learning_rate=0.1)
nn_mnist.debug_shapes = True

print(f"Network architecture:")
print(f"Input layer: 784 neurons (28×28 pixel values)")
print(f"Hidden layer: 128 neurons (sigmoid activation)")
print(f"Output layer: 10 neurons (softmax activation for digits 0-9)")
print(f"Total parameters: {784*128 + 128 + 128*10 + 10:,}")

losses, accuracies = nn_mnist.train(X_train_proc, y_train_onehot, epochs=500, print_every=40, batch_size=128)

print(f"\nEvaluating on test set...")
test_predictions = nn_mnist.forward(X_test_proc)
test_accuracy = nn_mnist.compute_accuracy(y_test_onehot, test_predictions)
print(f"Final Test Accuracy: {test_accuracy:.4f}")

# ------------------------------------------------------------------------------------------------------------------------------------------------

model_mnist = keras.Sequential([
    keras.layers.Dense(128, activation='sigmoid', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

model_mnist.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

print("TensorFlow model architecture:")
model_mnist.summary()

print("\nTraining TensorFlow model...")
history = model_mnist.fit(X_train_proc, y_train_onehot,
                         epochs=20,
                         batch_size=128,
                         validation_split=0.2,
                         verbose=1)

test_loss, test_acc = model_mnist.evaluate(X_test_proc, y_test_onehot, verbose=0)
print(f"\nTensorFlow Test Accuracy: {test_acc:.4f}")

# ------------------------------------------------------------------------------------------------------------------------------------------------

# Plot training progress
plt.figure(figsize=(15, 5))

# From-scratch results
plt.subplot(1, 3, 1)
plt.plot(losses)
plt.title('From-Scratch NN: Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(accuracies)
plt.title('From-Scratch NN: Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

# TensorFlow results
plt.subplot(1, 3, 3)
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('TensorFlow NN: Training History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------------------


def test_predictions(model_func, X_test, y_test, n_examples=10):
    """Test predictions on sample images"""
    # Get predictions
    if hasattr(model_func, 'predict'):  # TensorFlow model
        predictions = model_func.predict(X_test[:n_examples])
        pred_labels = np.argmax(predictions, axis=1)
    else:  # Our from-scratch model
        predictions = model_func.forward(X_test[:n_examples])
        pred_labels = np.argmax(predictions, axis=1)
    
    true_labels = np.argmax(y_test[:n_examples], axis=1)
    
    # Plot results
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(n_examples):
        row = i // 5
        col = i % 5
        
        # Reshape back to 28x28 for visualization
        image = X_test[i].reshape(28, 28)
        axes[row, col].imshow(image, cmap='gray')
        
        # Color code: green if correct, red if wrong
        color = 'green' if pred_labels[i] == true_labels[i] else 'red'
        axes[row, col].set_title(f'True: {true_labels[i]}, Pred: {pred_labels[i]}', 
                                color=color)
        axes[row, col].axis('off')
    
    return pred_labels, true_labels

print("\n" + "="*60)
print("TESTING PREDICTIONS ON SAMPLE IMAGES")
print("="*60)

print("From-scratch neural network predictions:")
pred_scratch, true_labels = test_predictions(nn_mnist, X_test_proc, y_test_onehot)
plt.suptitle('From-Scratch NN Predictions (Green=Correct, Red=Wrong)')
plt.show()

print("\nTensorFlow neural network predictions:")
pred_tf, _ = test_predictions(model_mnist, X_test_proc, y_test_onehot)
plt.suptitle('TensorFlow NN Predictions (Green=Correct, Red=Wrong)')
plt.show()