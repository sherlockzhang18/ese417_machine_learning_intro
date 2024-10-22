import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

# (1)
n = 50

X, y = datasets.make_blobs(n_samples=n, centers=2, n_features=2, center_box=(0, 10), cluster_std=1, random_state=1247)
y = np.where(y == 0, -1, 1)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', marker='o', edgecolor='k')
plt.title('Linearly Separable Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# (2)
def batch_perceptron(X_train, y_train, eta=0.01, epochs=100):
    w = np.array([0,0])
    b = 0
    
    errors = []
    
    for epoch in range(epochs):
        misclassified = []
        error_count = 0
        
        for i in range(X_train.shape[0]):
            if y_train[i] * (np.dot(X_train[i], w) + b) <= 0:
                misclassified.append((X_train[i], y_train[i]))
                error_count += 1
                
        for x_i, y_i in misclassified:
            w = w + eta * y_i * x_i
            b = b + eta * y_i
        
        errors.append(error_count)
        
        if error_count == 0:
            break
    
    return w, b, errors


w, b, errors = batch_perceptron(X_train, y_train, eta=0.01, epochs=100)

plt.plot(errors)
plt.xlabel('Epochs')
plt.ylabel('Number of Misclassified Points')
plt.title('Error Function Curve')

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0], colors='black')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
plt.title('Decision Boundary and Training Points')

# (3)

def perceptron_predict(X, w, b):
    return np.where(np.dot(X, w) + b > 0, 1, -1)

y_pred = perceptron_predict(X_test, w, b)

accuracy = np.mean(y_pred == y_test)
error_rate = 1 - accuracy

print(f"Batch Perceptron Accuracy: {accuracy * 100:.2f}%")
print(f"Batch Perceptron Error rate: {error_rate * 100:.2f}%")

plt.contour(xx, yy, Z, levels=[0], colors='black')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.Paired)
plt.title('Decision Boundary and Test Points')

# (4)

def sequential_perceptron(X_train, y_train, eta=0.01, epochs=100):
    w = np.array([-1, 1])
    b = 0
    weights = []
    
    for epoch in range(epochs):
        for i in range(X_train.shape[0]):
            if y_train[i] * (np.dot(X_train[i], w) + b) <= 0:
                w = w + eta * y_train[i] * X_train[i]
                b = b + eta * y_train[i]
                weights.append((w.copy(), b))
        
        predictions = np.sign(np.dot(X_train, w) + b)
        if np.all(predictions == y_train):
            break
    
    return w, b, weights

w_seq, b_seq, weight_history = sequential_perceptron(X_train, y_train, eta=0.01, epochs=100)

weights_array = np.array([w for w, _ in weight_history])
plt.plot(weights_array[:, 0], label="Weight 1")
plt.plot(weights_array[:, 1], label="Weight 2")
plt.xlabel('Iterations')
plt.ylabel('Weight Values')
plt.title('Weights vs Iterations (Sequential Perceptron)')
plt.legend()

Z_seq = np.dot(np.c_[xx.ravel(), yy.ravel()], w_seq) + b_seq
Z_seq = Z_seq.reshape(xx.shape)

plt.contour(xx, yy, Z_seq, levels=[0], colors='black')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
plt.title('Decision Boundary (Sequential Perceptron) and Training Points')


# (5)
y_pred_seq = perceptron_predict(X_test, w_seq, b_seq)

accuracy_seq = np.mean(y_pred_seq == y_test)
error_rate_seq = 1 - accuracy_seq

print(f"Sequential Perceptron Accuracy: {accuracy_seq * 100:.2f}%")
print(f"Sequential Perceptron Error Rate: {error_rate_seq * 100:.2f}%")

plt.contour(xx, yy, Z_seq, levels=[0], colors='black')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', cmap=plt.cm.Paired)
plt.title('Decision Boundary (Sequential Perceptron) and Test Points')

# (6)