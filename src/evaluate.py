import numpy as np
from model import forward_propagation
from preprocess import preprocess_data

# Load preprocessed data
X, y = preprocess_data('data/titanic.csv')

# Load the saved trained parameters
weights_input_hidden = np.load('weights_input_hidden.npy')
weights_hidden_output = np.load('weights_hidden_output.npy')
bias_hidden = np.load('bias_hidden.npy')
bias_output = np.load('bias_output.npy')

# Split data into training and testing sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate the model
hidden_output_test, final_output_test = forward_propagation(X_test, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

# Convert the output to predictions (0 or 1)
predictions = (final_output_test > 0.5).astype(int)  # 0 if not survived, 1 if survived

# Calculate accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy on test set: {accuracy * 100:.2f}%")
