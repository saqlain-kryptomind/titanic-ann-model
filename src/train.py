import numpy as np
from preprocess import preprocess_data
from model import initialize_parameters, forward_propagation, backpropagation

# Load and preprocess data
X, y = preprocess_data('data/titanic.csv')

# Split data into training and testing sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize neural network parameters
input_neurons = X_train.shape[1]
hidden_neurons = 3
output_neurons = 1

weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = initialize_parameters(input_neurons, hidden_neurons, output_neurons)

# Train the model
epochs = 20000
for epoch in range(epochs):
    hidden_output, final_output = forward_propagation(X_train, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
    weights_input_hidden, weights_hidden_output, bias_hidden, bias_output = backpropagation(X_train, y_train, hidden_output, final_output, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

    if epoch % 1000 == 0:
        loss = np.mean(np.square(y_train - final_output))
        print(f"Epoch {epoch}, Loss: {loss}")
        
# Save the trained parameters
np.save('weights_input_hidden.npy', weights_input_hidden)
np.save('weights_hidden_output.npy', weights_hidden_output)
np.save('bias_hidden.npy', bias_hidden)
np.save('bias_output.npy', bias_output)
