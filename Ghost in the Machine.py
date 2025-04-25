import numpy as np

class GhostInMachine:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights for the neural network
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        
        # Learning rate and momentum
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.prev_weights_input_hidden_change = np.zeros_like(self.weights_input_hidden)
        self.prev_weights_hidden_output_change = np.zeros_like(self.weights_hidden_output)
    
    def forward(self, input_data):
        hidden_input = np.dot(input_data, self.weights_input_hidden)
        hidden_output = self.sigmoid(hidden_input)
        
        output_input = np.dot(hidden_output, self.weights_hidden_output)
        output = self.sigmoid(output_input)
        
        return output
    
    def backward(self, input_data, target, predicted):
        # Compute the error
        output_error = (predicted - target) * predicted * (1 - predicted)
        
        # Backpropagate the error to hidden layer
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * hidden_output * (1 - hidden_output)
        
        # Update weights for the hidden-output connection
        delta_weights_hidden_output = np.outer(hidden_output, output_error)
        self.weights_hidden_output -= self.learning_rate * delta_weights_hidden_output + self.momentum * self.prev_weights_hidden_output_change
        self.prev_weights_hidden_output_change = delta_weights_hidden_output
        
        # Update weights for the input-hidden connection
        delta_weights_input_hidden = np.outer(input_data, hidden_error)
        self.weights_input_hidden -= self.learning_rate * delta_weights_input_hidden + self.momentum * self.prev_weights_input_hidden_change
        self.prev_weights_input_hidden_change = delta_weights_input_hidden
    
    def train(self, data, epochs):
        for epoch in range(epochs):
            for input_data, target in data:
                predicted = self.forward(input_data)
                self.backward(input_data, target, predicted)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
# Example usage
input_size = 2
hidden_size = 3
output_size = 1

# Create the Ghost in the Machine
ghost = GhostInMachine(input_size, hidden_size, output_size)

# Generate some training data (XOR problem for simplicity)
data = [
    (np.array([0, 0]), np.array([0])),
    (np.array([0, 1]), np.array([1])),
    (np.array([1, 0]), np.array([1])),
    (np.array([1, 1]), np.array([0]))
]

# Train the ghost
epochs = 10000
ghost.train(data, epochs)

# Test the trained model
test_data = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
for input_data in test_data:
    output = ghost.forward(input_data)
    print(f"Input: {input_data} -> Output: {output}")
