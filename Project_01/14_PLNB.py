# Group Number   : 14
# Roll Numbers   : 21CS10042 - Meduri Harshith Chowdary, 23CS60R30 - Dangodra Gautam Bharat Geeta
# Project Number : PLNB
# Project Title  : Personal Loan acquisition using Naive Bayes Classifier based Learning Model

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier # sklearn Decision Tree
from sklearn.naive_bayes import MultinomialNB # sklearn Naive Bayes
from sklearn.naive_bayes import GaussianNB # sklearn Naive Bayes
import operator
from math import log
from collections import Counter

# Load the dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    header = df.columns.tolist()
    data = df.values.tolist()
    return header, data

# Convert string columns to numeric values
def preprocess_data(header, data):
    numeric_data = []
    ind = header.index('Personal Loan')
    for row in data:
        numeric_row = []
        for i in range(len(row)):
            if i == ind:
                continue
            else:
                try:
                    numeric_row.append(int(row[i]))
                except ValueError:
                    numeric_row.append(float(row[i]))
        
        numeric_row.append(int(row[ind]))
        
        numeric_data.append(numeric_row)
    return numeric_data

# Split the dataset into training and testing sets using train_test_split
def custom_train_test_split(data, test_size=0.2, random_state=None):
    if(test_size >= 0.999):
        print("Caution: test_size % must be between 0 % and 99.8 % for better results.")
        raise ValueError("test_size % must be less than or equal to 99.8 %")

    np.random.seed(random_state)
    np.random.shuffle(data)

    X = np.array(data)[:, :-1]  # Features
    y = np.array(data)[:, -1]  # Target variable

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Gaussian Naive Bayes Classifier
class CustomGaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {c: np.mean(X[y == c], axis=0) for c in self.classes}
        self.std = {c: np.std(X[y == c], axis=0) + 1e-10 for c in self.classes}  # Added small value to prevent division by zero
        self.priors = {c: len(y[y == c]) / len(y) for c in self.classes}

    def predict(self, X):
        posteriors = {c: np.sum(np.log(self.pdf(X, c)), axis=1) + log(self.priors[c]) for c in self.classes}
        return np.array([max(posteriors.keys(), key=lambda c: posteriors[c][i]) for i in range(len(X))])

    def pdf(self, X, c):
        mean = self.mean[c]
        std = self.std[c]
        pdf_values = np.exp(-0.5 * ((X - mean) / std) ** 2) / (np.sqrt(2 * np.pi) * std)
        pdf_values[pdf_values == 0] = 1e-10  # Replace zero values with a small value
        return pdf_values


# Evaluation metrics
def custom_accuracy(y_true, y_pred):
    correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    return correct / len(y_true)

def precision_recall_f1(y_true, y_pred, label=1):
    true_positive = sum((y_t == label) and (y_p == label) for y_t, y_p in zip(y_true, y_pred))
    false_positive = sum((y_t != label) and (y_p == label) for y_t, y_p in zip(y_true, y_pred))
    false_negative = sum((y_t == label) and (y_p != label) for y_t, y_p in zip(y_true, y_pred))

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    if precision + recall != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return precision, recall, f1_score

def process_classification_report(report_dict):
    float_array = []

    # Iterate through the classification report dictionary
    for label, metrics in report_dict.items():
        # Exclude the 'macro avg', 'weighted avg', and 'support' keys
        if label not in ['macro avg', 'weighted avg']:
            # Check if the label is 'accuracy' and append its value
            if label == 'accuracy':
                float_array.append(float(metrics))
            else:
                # Iterate through the metrics for each label
                for metric, value in metrics.items():
                    # Convert the value to float and append to the float_array
                    if metric != 'support':
                        float_array.append(float(value))

    return float_array

def compare(list1, list2, roundup):
    for i in range(len(list1)):
        if round(list1[i], roundup) != round(list2[i], roundup):
            return False
    return True


# Load dataset
header, data = load_dataset('loan.csv')  # Assuming the file is named 'loan.csv'

# Preprocess data
numeric_data = preprocess_data(header, data)

# Split dataset
test_size = float(input("\nEnter the test_size per-cent (between 0 & 100) : "))
X_train, X_test, y_train, y_test = custom_train_test_split(numeric_data, test_size/100, random_state=42)
# X_train, X_test, y_train, y_test = train_test_split(numeric_data, test_size/100, random_state=42)

### Gaussian Naive Bayes ###

# Train our Custom Gaussian Naive Bayes classifier
gnb_custom = CustomGaussianNaiveBayes()
gnb_custom.fit(X_train, y_train)

# Make predictions
y_pred_custom = gnb_custom.predict(X_test)

roundup = 3  # Number of decimal places to round to / Error tolerance for comparison of results (classification report)

# Evaluation metrics
acc_custom = accuracy_score(y_test, y_pred_custom)
report_custom = classification_report(y_test, y_pred_custom)
report_custom_dict = classification_report(y_test, y_pred_custom, output_dict=True)
processed_custom_report = process_classification_report(report_custom_dict)

# acc_custom = custom_accuracy(y_test, y_pred_custom)
# precision_custom, recall_custom, f1_custom = precision_recall_f1(y_test, y_pred_custom) # To use custom made precision, recall and f1_score function

# Train scikit-learn's Gaussian Naive Bayes classifier
gnb_sklearn = GaussianNB()
gnb_sklearn.fit(X_train, y_train)
y_pred_sklearn = gnb_sklearn.predict(X_test)

# Evaluation metrics for scikit-learn's Gaussian Naive Bayes
acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
report_sklearn = classification_report(y_test, y_pred_sklearn)
report_sklearn_dict = classification_report(y_test, y_pred_sklearn, output_dict=True)
processed_sklearn_report = process_classification_report(report_sklearn_dict)

print("\nTraing and Testing Data Split :", "{:.2f} %".format(100-test_size), "  and  ", "{:.2f} %".format(test_size))

# Print results - Comparison between ours and scikit-learn's Gaussian Naive Bayes
print("\nClassification Report (our):")
print(report_custom)

print("\nClassification Report (sklearn):")
print(report_sklearn)

# Check if all values under each label match within the specified tolerance
print("\nComparison of Results : [Match] => (True \ False) ? \n")
print("[Res] Custom <v/s> scikit-learn  [Overall]  : ", compare(processed_custom_report, processed_sklearn_report, roundup-1), " (upto", roundup, "decimal places)")
print("[Res] Custom <v/s> scikit-learn [Accuracy]  : ", abs(acc_custom - acc_sklearn) <= 0.01, " (upto 2 decimal places)")