import numpy as np

class NeuralNetwork:
    def __init__(self, learning_rate, input_size, hidden_size, output_size):
        # Initialise a neural network

        self.lr = learning_rate


        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.B1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size ) * 0.01
        self.B2 = np.zeros((1, output_size))

    def sigmoid(self,z):
        # Takes any number and squishes it into a value between 0 and 1

        z = np.clip(z, -500, 500) # Clipping the Z value prevents overflow
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self,z):
        s = self.sigmoid(z)
        return s * (1-s)
    
    def softmax(self,z):
        # softmax allows for multi-class classification
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward(self, inputData):
        # Forward propagation, input data moves through each layer of the neural network

        # Layer 1   -   Following Z = WX + B 
        self.Z1 = np.dot(inputData, self.W1) + self.B1
        self.A1 = self.sigmoid(self.Z1)

        # Layer 2 (output)  -  Using softmax for multi-class classification
        self.Z2 = np.dot(self.A1, self.W2) + self.B2
        self.A2 = self.softmax(self.Z2)

        return self.A2
    
    def backward(self, inputData, trueLabels, output):

        m = inputData.shape[0]

        # Calculate the gradients for the output layer
        # negative -> prediction too low, increase output. vice versa
        # This calculates how much each connection contributed

        dZ2 = output - trueLabels
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True) # Each gradient tells us how to adjust the weight from that neuron to the output

        # Calculate the gradients for the hidden layer

        dZ1 = np.dot(dZ2, self.W2.T) * self.sigmoid_derivative(self.Z1)
        dW1 = (1/m) * np.dot(inputData.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update the weights and biases
        self.W2 -= self.lr * dW2
        self.B2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.B1 -= self.lr * db1

    def train(self, inputData, trueLabels, epochs = 1000, print_every=10):
        
        losses = []
        accuracies = []

        n_samples = inputData.shape[0]

        for epoch in range(epochs):

            # Perform a forward propagation

            output = self.forward(inputData)

            # Calculate loss

            loss = self.compute_loss(trueLabels, output)
            losses.append(loss)

            # Calculate accuracy

            accuracy = self.compute_accuracy(trueLabels, output)
            accuracies.append(accuracy)

            # Perform a backwards propagation

            self.backward(inputData, trueLabels, output)

            if epoch % print_every == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

        
        return losses, accuracies

    def compute_loss(self, trueLabels, output):
        # Computing binary cross-entropy loss
            
        m = trueLabels.shape[0]

        output = np.clip(output, 1e-7, 1-1e-7)
        loss = -1/m * np.sum(trueLabels * np.log(output))
            
        return loss
    
    def compute_accuracy(self, y_true, y_pred):

        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        return np.mean(y_true_labels == y_pred_labels)
    
    def predict(self, inputData):
        # Make the predictions using the input data
        output = self.forward(inputData)
        return (output > 0.5).astype(int)
