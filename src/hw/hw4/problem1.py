import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score

# Part (a):
x, y = make_moons(n_samples=200, noise=0.2, random_state=42)

plt.figure(figsize=(8, 6))
plt.scatter(x[y == 0, 0], x[y == 0, 1], color='red', label='Class 0')
plt.scatter(x[y == 1, 0], x[y == 1, 1], color='blue', label='Class 1')
plt.legend()
plt.title('Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Part (b): Train SVM with RBF kernel and different C values
def plot_decision_boundary(model, x, y, title):
    # Set min and max values and add some padding
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.02
    # Generate a grid of points
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Predict for the grid points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot contour and data points
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=ListedColormap(('red', 'blue')))
    plt.scatter(x[y == 0, 0], x[y == 0, 1], color='red', label='Class 0')
    plt.scatter(x[y == 1, 0], x[y == 1, 1], color='blue', label='Class 1')
    # Plot support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=100, facecolors='none', edgecolors='k', label='Support Vectors')
    plt.legend()
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

C_values = [0.05, 0.1, 1, 2, 5, 10, 50, 100]

for C in C_values:
    # Train the SVM model with RBF kernel and parameter C
    svm_rbf = SVC(kernel='rbf', C=C)
    svm_rbf.fit(x_train, y_train)
    # Plot the decision boundary and support vectors
    title = f'SVM with RBF kernel and C={C}'
    plot_decision_boundary(svm_rbf, x_train, y_train, title)

# Part (b) Explanation:
# Smaller values of C allow for a wider margin, potentially misclassifying some points
# but aiming for better generalization. Larger values of C try to classify all training
# points correctly, which may lead to overfitting.

# Part (c): Train models using different kernels and evaluate
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
test_accuracies = {}

for kernel in kernels:
    # Train the SVM model with the specified kernel
    svm_model = SVC(kernel=kernel, C=1, gamma='auto')
    svm_model.fit(x_train, y_train)
    # Plot the decision boundary
    title = f'SVM with {kernel} kernel'
    plot_decision_boundary(svm_model, x_train, y_train, title)
    # Predict on the test set
    y_pred = svm_model.predict(x_test)
    # Calculate test accuracy
    accuracy = accuracy_score(y_test, y_pred)
    test_accuracies[kernel] = accuracy
    print(f"Test accuracy with {kernel} kernel: {accuracy:.2f}")

# Determine which kernel gives the best performance
best_kernel = max(test_accuracies, key=test_accuracies.get)
print(f"The kernel with the best performance on the test dataset is: '{best_kernel}'")