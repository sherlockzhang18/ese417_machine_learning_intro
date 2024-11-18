import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-hot encode the labels
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# Train-test split
X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42
)
_, _, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Activation functions and their derivatives
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    N = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred)) / N
    return loss

def compute_accuracy(y_true, y_pred):
    predictions = np.argmax(y_pred, axis=1)
    labels = np.argmax(y_true, axis=1)
    accuracy = np.mean(predictions == labels)
    return accuracy

def initialize_parameters(n_input, n_hidden, n_output):
    np.random.seed(42)
    W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2. / n_input)
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2. / n_hidden)
    b2 = np.zeros((1, n_output))
    return W1, b1, W2, b2

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    cache = (Z1, A1, Z2, A2)
    return A2, cache

def backward_propagation(X, y_true, W1, b1, W2, b2, cache):
    Z1, A1, Z2, A2 = cache
    N = X.shape[0]
    dZ2 = A2 - y_true
    dW2 = np.dot(A1.T, dZ2) / N
    db2 = np.sum(dZ2, axis=0, keepdims=True) / N
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / N
    db1 = np.sum(dZ1, axis=0, keepdims=True) / N
    return dW1, db1, dW2, db2

def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

def train_mlp(X_train, y_train, X_val, y_val, n_hidden=64, epochs=1000, learning_rate=0.01):
    n_input = X_train.shape[1]
    n_output = y_train.shape[1]
    W1, b1, W2, b2 = initialize_parameters(n_input, n_hidden, n_output)
    losses = []
    accuracies = []
    for epoch in range(epochs):
        A2, cache = forward_propagation(X_train, W1, b1, W2, b2)
        loss = cross_entropy_loss(y_train, A2)
        losses.append(loss)
        dW1, db1, dW2, db2 = backward_propagation(X_train, y_train, W1, b1, W2, b2, cache)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if (epoch + 1) % 100 == 0 or epoch == 0:
            A2_val, _ = forward_propagation(X_val, W1, b1, W2, b2)
            val_accuracy = compute_accuracy(y_val, A2_val)
            accuracies.append(val_accuracy)
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Validation Accuracy: {val_accuracy:.4f}")
    return W1, b1, W2, b2, losses, accuracies

def predict(X, W1, b1, W2, b2):
    A2, _ = forward_propagation(X, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=1)
    return predictions

# Experiment with different numbers of hidden nodes
hidden_node_counts = [10, 20, 30, 40, 50, 64, 80, 100]
test_accuracies = []

for n_hidden in hidden_node_counts:
    print(f"\nTraining MLP with {n_hidden} hidden nodes.")
    W1, b1, W2, b2, losses, val_accuracies = train_mlp(
        X_train, y_train_encoded, X_test, y_test_encoded,
        n_hidden=n_hidden, epochs=500, learning_rate=0.01
    )
    y_test_pred = predict(X_test, W1, b1, W2, b2)
    test_accuracy = np.mean(y_test_pred == y_test)
    test_accuracies.append(test_accuracy)
    print(f"Test Accuracy with {n_hidden} hidden nodes: {test_accuracy:.4f}")

# Plot the test accuracies
plt.figure(figsize=(10, 6))
plt.plot(hidden_node_counts, test_accuracies, marker='o')
plt.title('Test Accuracy vs. Number of Hidden Nodes')
plt.xlabel('Number of Hidden Nodes')
plt.ylabel('Test Accuracy')
plt.grid(True)
plt.show()

# Find the best number of hidden nodes
best_accuracy = max(test_accuracies)
best_n_hidden = hidden_node_counts[test_accuracies.index(best_accuracy)]
print(f"\nThe best number of hidden nodes is {best_n_hidden} with a test accuracy of {best_accuracy:.4f}")
