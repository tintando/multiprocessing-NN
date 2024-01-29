import numpy as np
from sklearn.datasets import make_regression

# Generate a regression dataset with 100 samples, 5 input features, and 2 output targets
X, y = make_regression(n_samples=100, n_features=5, n_targets=2)

# Print the shape of the input and output arrays
print("Input shape:", X.shape)
print("Output shape:", y.shape)

class MLP:
    def __init__(self):
        self.layers = []
        self.weights = []
        self.biases = []
    
    def add_layer(self, size):
        self.layers.append(size)
        if len(self.layers) > 1:
            prev_size = self.layers[-2]
            curr_size = self.layers[-1]
            w = np.random.randn(prev_size, curr_size)
            self.weights.append(w)
    
    def forward(self, x):
        print([w.shape for w in self.weights])
        a = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = self.sigmoid(z)
        return a
    
    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, x, y, learning_rate):
        # Perform forward propagation
        output = self.forward(x)
        
        # Compute the loss
        loss = self.compute_loss(output, y)
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros(w.shape[1]) for w in self.weights]
        
        # Backpropagate the gradients
        delta = output - y
        for i in range(len(self.weights)-1, -1, -1):
            dW[i] = np.dot(self.layers[i].T, delta)
            db[i] = np.sum(delta, axis=0)
            delta = np.dot(delta, self.weights[i].T) * self.layers[i] * (1 - self.layers[i])
        
        # Update the weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * db[i]
        
        return loss
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)


# Create an instance of the MLP class
mlp = MLP()

# Add layers to the MLP
input_size = 10
mlp.add_layer(input_size)  # Input layer with 10 neurons
mlp.add_layer(20)  # Hidden layer with 20 neurons
mlp.add_layer(5)   # Output layer with 5 neurons

# Define an input vector
x = np.random.randn(input_size)

# Perform forward propagation
output = mlp.forward(x)

# Print the output
print(output)
