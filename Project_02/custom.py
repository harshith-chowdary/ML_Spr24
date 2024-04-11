import numpy as np
import pandas as pd
import sys

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#############################################################################################################

# Calculating accuracy, total number of correct predictions / total number of predictions
def calculate_accuracy(predictions, targets):
    correct = np.sum(predictions == targets)
    total = len(targets)
    accuracy = correct / total
    return accuracy

# Sigmoid activation function
def sigmoid(x):
    # Avoid overflow and underflow issues by clipping the values
    clipped_x = np.clip(x, -500, 500)
    return np.where(clipped_x >= 500, 1.0, np.where(clipped_x <= -500, 0.0, 1 / (1 + np.exp(-clipped_x))))

# Derivative of sigmoid function for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# ReLU activation function
def relu(x):
    return np.maximum(0, x)

# Derivative of ReLU function for backpropagation
def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Cross entropy loss function
def compute_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # to avoid division by zero
    return (y_true - y_pred)/(y_pred * (1 - y_pred))

# Binary cross entropy loss function
def binary_cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-10  # Small value to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip values to avoid log(0)
    loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss / len(y_true)  # Normalize by number of samples

# loss function: categorical cross entropy loss, for backpropagation, we need the average of derivative of cross entropy loss function for the batch
def categorical_cross_entropy(y_true, y_pred):
    n_samples = y_true.shape[0]
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    loss = 0
    ind = 0
    for i in range(n_samples):
        tem = (y_true[i] - y_pred[ind])/(y_pred[ind] * (1 - y_pred[ind]))
        loss -= tem
        ind += 1
    loss = loss / n_samples
    return loss

# Initialising weights randomly
def initialize_weights(layers):
    weights = []
    
    for i in range(len(layers) - 1):
        w = np.random.uniform(-1, 1, (layers[i], layers[i+1]))
        weights.append(w)
        
        print(w.shape)
        
    return weights

# Forward pass through the network for one layer
def forward(inputs, weights): 
    outputs = []
    layer_output = inputs
    
    for i in range(len(weights)-1):
        layer_input = np.dot(layer_output, weights[i])
        layer_output = sigmoid(layer_input)
            
        outputs.append(layer_output)
        
    output = np.dot(layer_output, weights[-1])
    output = sigmoid(output)
    # output = relu(output)
    
    outputs.append(output)

    return outputs

# Backward pass through the network for one layer
def backward(inputs, outputs, targets, weights, learning_rate):
    targets = targets.to_numpy().reshape(-1, 1)
    # loss = targets - output # loss function: squared error loss
    # loss = categorical_cross_entropy(targets, output) # loss function: categorical cross entropy loss
    loss = compute_loss(targets, outputs[-1]) # loss function: binary cross entropy loss
    loss = loss/len(targets)

    for i in range(len(weights)-1, -1, -1):
        # Calculate the gradient of the loss function w.r.t the output
        if i == len(weights)-1:
            grad_delta = loss * sigmoid_derivative(outputs[i])
            # grad_delta = loss * relu_derivative(outputs[i])
        else:
            grad_delta = np.dot(grad_delta, weights[i+1].T) * sigmoid_derivative(outputs[i])

        # Calculate the gradient of the loss function w.r.t the weights
        if i == 0:
            grad_weights = np.dot(inputs.T, grad_delta)
        else:
            grad_weights = np.dot(outputs[i-1].T, grad_delta)
        
        # Update the weights
        weights[i] += learning_rate * grad_weights
        
    return weights

def train_neural_network(data_train, data_test, layers, epochs, learning_rate, batch_size=128):
    weights = initialize_weights(layers)
    
    for epoch in range(epochs):
        print(f"\t\t\t\t\t\tEpoch [{epoch+1}/ {epochs}]")
        
        num_batches = len(data_train) // batch_size
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size
            
            data_batch = data_train[start:end]
            
            X_batch = data_batch.drop('hazardous', axis=1)
            y_batch = data_batch['hazardous']
            
            outputs = forward(X_batch, weights)
            
            weights = backward(X_batch, outputs, y_batch, weights, learning_rate)
            
            # if batch_idx % 100 == 0:
            #     print(f"Batch {batch_idx}/{num_batches} => {start}:{end}")
                
            #     train_pred = predict(data_train, weights)
            #     train_accuracy = calculate_accuracy(train_pred, data_train['hazardous'])
                
            #     print(f"Training accuracy till now : {100 * train_accuracy:>0.1f}%")
        
        if epoch % 10 == 9:
            # Calculate training accuracy
            train_pred = predict(data_train, weights)
            train_accuracy = calculate_accuracy(train_pred, data_train['hazardous'])
            print(f"Training accuracy @ Epoch {epoch+1}/{epochs} : {100 * train_accuracy:>0.1f}%")
            
    return weights

# Prediction function, getting the predictions from the trained model
def predict(test_data, weights):
    predictions = []
    
    test_data = test_data.to_numpy()
    
    for inputs in test_data:
        outputs = forward(inputs[:-1], weights)
        output = outputs[-1]
        
        predictions.append(1 if output >= 0.5 else 0)
    return np.array(predictions)

#############################################################################################################

# Load the data
data = pd.read_csv('neo.csv')

data.drop('id', axis=1, inplace=True)
data.drop('name', axis=1, inplace=True)
data.drop('orbiting_body', axis=1, inplace=True)
data.drop('sentry_object', axis=1, inplace=True)

# Assuming 'data' is your pandas DataFrame
data['est_diameter_min'] = data['est_diameter_min'].astype(float)
data['est_diameter_max'] = data['est_diameter_max'].astype(float)
data['relative_velocity'] = data['relative_velocity'].astype(float)
data['miss_distance'] = data['miss_distance'].astype(float)
data['absolute_magnitude'] = data['absolute_magnitude'].astype(float)
    
# Mapping the output into binary classes
data['hazardous'] = data['hazardous'].map({True: 1, False: 0})

data = shuffle(data, random_state=0)

# Preprocess the data
data_train, data_test = train_test_split(data, test_size=0.2, random_state=4)

# Normalising the data
X_data_train = data_train.drop('hazardous', axis=1)
Y_data_train = data_train['hazardous']

X_data_test = data_test.drop('hazardous', axis=1)
Y_data_test = data_test['hazardous']

X_data_train = (X_data_train - X_data_train.mean()) / X_data_train.std()
X_data_test = (X_data_test - X_data_test.mean()) / X_data_test.std()

data_train = pd.concat([X_data_train, Y_data_train], axis=1)
data_test = pd.concat([X_data_test, Y_data_test], axis=1)

# Inserting a column of 1s for the constant term for the w0 term
data_train.insert(0, 'const', 1)
data_test.insert(0, 'const', 1)

# Train the neural network
input_size = data_train.shape[1] - 1

hidden_size_i = [] # No hidden layers
hidden_size_ii = [16] # 16 neurons in the hidden layer
hidden_size_iii = [32] # 32 neurons in the hidden layer

hidden = [hidden_size_i, hidden_size_ii, hidden_size_iii]

output_size = 1
batch_size = 128

num_epochs = 4

# learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]

type = int(sys.argv[1])

learning_rate = float(sys.argv[2])

layers = [input_size] + hidden[type-1] + [output_size]

print(layers)

print("Custom Neural Network with ", end="")

if(type == 1):
    print("No hidden layers")
    
elif(type == 2):
    print("16 neurons in the hidden layer")
    
else:
    print("32 neurons in the hidden layer")

# for learning_rate in learning_rates:
print("\nLearning rate of the Model :", learning_rate)

# Predict on the test set
predictions = predict(data_test, train_neural_network(data_train, data_test, layers, num_epochs, learning_rate, batch_size))
accuracy = calculate_accuracy(predictions, data_test['hazardous'])
print('\nTesting accuracy:', accuracy, end="\n\n")

print(classification_report(data_test['hazardous'], predictions, zero_division=1))

print("---------------------------------------------------------------------------------")