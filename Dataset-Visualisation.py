import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as ts
from tensorflow import keras

def visualise_mnist_dataset():

    # display samples of every digit

    for digit in range(10):
        digit_indices = np.where(y_train == digit)[0][:8]

        for i, idx in enumerate(digit_indices):
            plt.subplot(10,8, digit * 8 + i + 1)
            plt.imshow(X_train[idx], cmap='gray', interpolation='nearest')
            if i == 0:
                plt.ylabel(digit, fontsize=12, fontweight='bold')
            plt.xticks([])
            plt.yticks([])
    plt.suptitle('MNIST Dataset : Examples of each digit (0-9)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Count the examples per each digit
    
    unique, counts = np.unique(y_train, return_counts=True)
    print("Training exmaples per digit:")
    for digit, count in zip(unique,counts):
        print(f"    Digit {digit}: {count:,} examples")

    # Plot class distribution

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(unique, counts, color='skyblue', edgecolor='navy')
    plt.title('Training Data Distribution')
    plt.xlabel('Digit')
    plt.ylabel('Number of Examples')
    plt.xticks(range(10))
    for i, count in enumerate(counts):
        plt.text(i, count + 100, str(count), ha='center', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    plt.subplot(1, 2, 2)

    # Calculate average pixel intensity for each digit

    avg_intensities = []
    for digit in range(10):
        digit_images = X_train[y_train == digit]
        avg_intensity = np.mean(digit_images)
        avg_intensities.append(avg_intensity)
    
    plt.bar(range(10), avg_intensities, color='lightcoral', edgecolor='darkred')
    plt.title('Average Pixel Intensity by Digit')
    plt.xlabel('Digit')
    plt.ylabel('Average Intensity (0-255)')
    plt.xticks(range(10))
    for i, intensity in enumerate(avg_intensities):
        plt.text(i, intensity + 2, f'{intensity:.1f}', ha='center', fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)

    # Sample a few thousand pixels from random images

    sample_pixels = X_train[:1000].flatten()
    plt.hist(sample_pixels, bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.title('Pixel Value Distribution (Raw Data)')
    plt.xlabel('Pixel Value (0-255)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)

    # Show normalized pixel distribution

    normalized_pixels = sample_pixels / 255.0
    plt.hist(normalized_pixels, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.title('Pixel Value Distribution (Normalized 0-1)')
    plt.xlabel('Pixel Value (0-1)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return avg_intensities