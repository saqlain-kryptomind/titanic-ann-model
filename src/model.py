import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define Binary Cross-Entropy Loss function
def binary_cross_entropy_loss(y_true, y_pred):
    # avoid log(0)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    # Calculate binary cross-entropy loss
    loss = - np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# Initialize ANN parameters (weights and biases)
def initialize_parameters(input_neurons, hidden_neurons, output_neurons):
    np.random.seed(42)
    weights_input_hidden = np.random.rand(input_neurons, hidden_neurons)
    weights_hidden_output = np.random.rand(hidden_neurons, output_neurons)
    bias_hidden = np.random.rand(hidden_neurons)
    bias_output = np.random.rand(output_neurons)
    
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output

# Forward propagation
def forward_propagation(X, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    # Input to hidden layer
    hidden_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_output = sigmoid(hidden_input)

    # Hidden to output layer
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = sigmoid(final_input)
    
    return hidden_output, final_output

# Backpropagation
def backpropagation(X, y, hidden_output, final_output, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output, learning_rate=0.1):
    output_error = y - final_output
    output_delta = output_error * sigmoid_derivative(final_output)

    hidden_error = np.dot(output_delta, weights_hidden_output.T) 
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output)

    weights_hidden_output += np.dot(hidden_output.T, output_delta) * learning_rate
    weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
    bias_output += np.sum(output_delta, axis=0) * learning_rate
    bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate
    
    return weights_input_hidden, weights_hidden_output, bias_hidden, bias_output
