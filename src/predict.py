import numpy as np
import pandas as pd
from preprocess import preprocess_data
from model import forward_propagation

def predict_new_data(new_data_file, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output):
    # Preprocess the new data
    new_data, _ = preprocess_data(new_data_file)

    # Make predictions using forward propagation
    _, predictions = forward_propagation(new_data, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)
    
    # Apply threshold (0.5) to convert sigmoid output into binary prediction (0 or 1)
    predictions = (predictions > 0.5).astype(int)
    
    return predictions

if __name__ == "__main__":
    # Load the trained model parameters (weights and biases)
    weights_input_hidden = np.load('weights_input_hidden.npy')
    weights_hidden_output = np.load('weights_hidden_output.npy')
    bias_hidden = np.load('bias_hidden.npy')
    bias_output = np.load('bias_output.npy')

    # File path to the new data (new_data.csv)
    new_data_file = 'new_data.csv'

    # Predict using the trained model
    predictions = predict_new_data(new_data_file, weights_input_hidden, weights_hidden_output, bias_hidden, bias_output)

    # Output the predictions
    print("Predictions (Survival):", predictions)
