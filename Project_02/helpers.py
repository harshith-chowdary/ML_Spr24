import numpy as np

def calculate_accuracy(predictions, targets):
    if type(targets) == list:
        targets = np.array(targets)
    
    correct = np.sum(predictions == targets)
    accuracy = correct / len(targets)
    
    return accuracy

def binary_cross_entropy_loss(predictions, targets):
    if type(targets) == list:
        targets = np.array(targets).reshape(-1, 1)
    
    epsilon = 1e-10
    
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    loss = -np.sum(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    
    return loss / len(targets)

def categorical_cross_entropy_loss(predictions, targets):
    if type(targets) == list:
        targets = np.array(targets).reshape(-1, 1)
    
    epsilon = 1e-15
    
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    ce_loss = -np.sum(targets * np.log(predictions))
    
    return ce_loss

# loss function: categorical cross entropy loss, for backpropagation, we need the average of derivative of cross entropy loss function for the batch
def categorical_cross_entropy(predictions, targets):
    if type(targets) == list:
        targets = np.array(targets).reshape(-1, 1)
    
    n_samples = targets.shape[0]
    predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
    loss = 0
    ind = 0
    for i in range(n_samples):
        tem = (targets[i] - predictions[ind])/(predictions[ind] * (1 - predictions[ind]))
        loss -= tem
        ind += 1
    loss = loss / n_samples
    return abs(loss)