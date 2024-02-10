PLNB: Personal Loan Acquisition using Naive Bayes Classifier

Code Overview
This project implements a Naive Bayes classifier for predicting personal loan acquisition. Below is an overview of the key components and functionalities of the code:

Libraries Used:
numpy: For numerical computations.
pandas: For data manipulation and CSV file I/O.
sklearn: For model training, evaluation, and splitting the dataset.
tabulate: For generating formatted tables.

Functions Implemented:
load_dataset(file_path): Loads the dataset from a CSV file and returns the header and data.

preprocess_data(header, data): Converts string columns to numeric values in the dataset.

custom_train_test_split(data, test_size=0.2, random_state=None): Splits the dataset into training and testing sets with a customizable test size.

CustomGaussianNaiveBayes: Implements a custom Gaussian Naive Bayes classifier with methods for fitting and predicting.

custom_accuracy(y_true, y_pred): Computes the accuracy score for model evaluation.

precision_recall_f1(y_true, y_pred, label=1): Computes precision, recall, and F1-score for a specified label.

process_classification_report(report_dict): Processes the classification report dictionary to extract relevant metrics.

compare(list1, list2, roundup): Compares two lists of metrics within a specified tolerance.

Workflow
Load the dataset and preprocess the data.
Split the dataset into training and testing sets.
Train a custom Gaussian Naive Bayes classifier on the training data.
Make predictions on the testing data using the custom classifier.
Evaluate the model's performance using accuracy and classification reports.
Train scikit-learn's Gaussian Naive Bayes classifier for comparison.
Compare the results of the custom and scikit-learn implementations.

Results
The code outputs classification reports for both the custom and scikit-learn implementations, along with a comparison of results.
The comparison ensures consistency between the custom implementation and scikit-learn's Gaussian Naive Bayes classifier.

Contributors
Meduri Harshith Chowdary (21CS10042)
Dangodra Gautam Bharat Geeta Dangodra (23CS60R30)