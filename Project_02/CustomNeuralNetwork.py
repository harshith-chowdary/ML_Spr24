import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from helpers import categorical_cross_entropy_loss, binary_cross_entropy_loss, calculate_accuracy

class CustomNeuralNetwork:
    def __init__(self, input_size, output_size, hidden_layers=[]):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.weights = []
        self.layer_outputs = []
        
        # Initialize weights and biases for each layer
        layer_sizes = [input_size] + hidden_layers + [output_size]
        
        for i in range(len(layer_sizes) - 1): 
            # Initialize weights using random normal distribution
            weight_matrix = np.random.uniform(-1, 1, (layer_sizes[i], layer_sizes[i+1])) 

            # Append initialized weights  to the lists
            self.weights.append(weight_matrix)
        
    def sigmoid(self, x):
        # Apply clipping to limit the range of input values
        clipped_x = np.clip(x, -500, 500)  # Adjust the clipping threshold as needed
        return 1 / (1 + np.exp(-clipped_x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def forward(self, X):
        self.layer_outputs = []
        
        layer_input = np.array(X)
        
        for i in range(len(self.hidden_layers)):
            layer_output = np.dot(layer_input, self.weights[i])
            
            layer_input = self.relu(layer_output)
            self.layer_outputs.append(layer_output)
            
        output = np.dot(layer_input, self.weights[-1])
        
        output = self.sigmoid(output)
        self.layer_outputs.append(output)
        
        return output
    
    def backward(self, X, y, learning_rate):
        X = np.array(X)
        # Adjust y to match the shape of the output layer
        y = np.array(y).reshape(-1, self.output_size)
        
        grad_output = (y - self.layer_outputs[-1]) * self.sigmoid_derivative(self.layer_outputs[-1])
        
        for i in range(len(self.hidden_layers), 0, -1):
            grad_input = np.dot(grad_output, self.weights[i].T) * self.relu_derivative(self.layer_outputs[i-1])
            
            print(self.weights[i].head())
            
            self.weights[i] += (np.dot(self.layer_outputs[i-1].T, grad_output) * learning_rate)
            
            print(self.weights[i].head())
            
            grad_output = grad_input
           
        self.weights[0] += np.dot(X.T, grad_output) * learning_rate
        
    def train(self, X_train, y_train, loss_fn, epochs, learning_rate, batch_size):
        training_accuracy = []
        training_loss = []
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Shuffle the data and labels together
            # X_train, y_train = shuffle(X_train, y_train, random_state=42)
            
            predictions = []
            
            num_batches = len(X_train) // batch_size
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = (batch_idx + 1) * batch_size
                
                X_batch = X_train[start:end]
                y_batch = y_train[start:end]
                
                output = self.forward(X_batch)
                
                predictions.append(output)
                
                self.backward(X_batch, y_batch, learning_rate)
                
                if(batch_idx % 100 == 0):
                    for i in range(len(output)):
                        output[i] = 1 if output[i] >= 0.5 else 0
                        
                    training_accuracy.append(calculate_accuracy(output, y_batch))
                    loss = loss_fn(output, y_batch).item()
                    training_loss.append(loss)
                    
                    print(f"loss: {loss:>7f}  [{start:>5d}/{end:>5d}]")
                    
            accuracy = calculate_accuracy(np.concatenate(predictions), y_train)
            print(f"===> Accuracy: {100 * accuracy:>0.1f}%")
                    
        # return training_accuracy, training_loss
            
    def test(self, X_test, y_test, loss_fn):
        output = self.predict(X_test)
        loss = loss_fn(y_test, output)
        
        accuracy = calculate_accuracy(output, y_test)
        
        print(f"\nTest Error: \n\tAccuracy: {(100*accuracy):>0.1f}%, Avg loss: {loss:>8f} \n")
                    
    
    # Prediction function, getting the predictions from the trained model
    def predict(self, test_data):
        predictions = []
        test_data = np.array(test_data)
        
        # print("test_data", test_data.shape)
        for inputs in test_data:
            output = self.forward(np.array(inputs))
            predictions.append(1 if output >= 0.5 else 0)
        
        return np.array(predictions)