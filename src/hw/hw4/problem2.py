# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load the digits dataset
digits = load_digits()

# Features and labels
X = digits.data
y = digits.target

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a range of hidden layer sizes to evaluate
hidden_layer_sizes = [(n,) for n in range(10, 101, 10)]  # From 10 to 100 nodes

# Store results
test_accuracies = []
train_accuracies = []
best_accuracy = 0
best_hidden_size = None
best_model = None

# Loop over different hidden layer sizes
for size in hidden_layer_sizes:
    # Define the MLP classifier with the given hidden layer size
    mlp = MLPClassifier(hidden_layer_sizes=size, activation='relu', solver='adam',
                        max_iter=500, random_state=42)
    
    # Fit the model to the training data
    mlp.fit(X_train, y_train)
    
    # Evaluate on the training data
    y_train_pred = mlp.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_accuracies.append(train_accuracy)
    
    # Evaluate on the test data
    y_test_pred = mlp.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_accuracies.append(test_accuracy)
    
    # Check if this is the best model so far
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_hidden_size = size
        best_model = mlp
    
    print(f"Hidden layer size {size}: Train Accuracy = {train_accuracy:.4f}, Test Accuracy = {test_accuracy:.4f}")

# Plot the accuracies vs. hidden layer sizes
hidden_nodes = [size[0] for size in hidden_layer_sizes]
plt.figure(figsize=(12, 6))
plt.plot(hidden_nodes, train_accuracies, label='Training Accuracy')
plt.plot(hidden_nodes, test_accuracies, label='Testing Accuracy')
plt.xlabel('Number of Hidden Nodes')
plt.ylabel('Accuracy')
plt.title('MLP Classifier Performance on Digits Dataset')
plt.legend()
plt.grid(True)
plt.show()

# Output the best number of hidden nodes
print(f"\nThe best number of hidden nodes is: {best_hidden_size[0]} with a test accuracy of {best_accuracy:.4f}")
