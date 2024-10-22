import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# part(a)
np.random.seed(0)
n = 100
sigma = 2

x = np.linspace(-5,5,n)
y0 = 12*np.sin(x) + 0.5 * (x**2) + 2*x + 5
y = y0 + np.random.randn(n)*sigma

figure1 = plt.figure("Figure 1")
plt.scatter(x,y,c='blue', label = "Generated data")
plt.plot(x, y0, label = "true pattern", color='green')
plt.title("True pattern of generated dataset")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# part(b)
model = Pipeline([
    ("Polynomial", PolynomialFeatures()), 
    ("Linear Regression", LinearRegression())
])

# part(c)
X = x[:,np.newaxis]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# part(d), (e)combined
degrees = [1, 2, 5, 8, 12, 14, 16, 18, 20]
train_errors = []
test_errors = []

plt.figure("Figure 2")
plt.scatter(x_test, y_test, color="blue", label="Test data")

for degree in degrees:
    model.set_params(Polynomial__degree=degree)
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    x_test_sorted = np.sort(x_test, axis=0)
    y_test_sorted_pred = model.predict(x_test_sorted)
    plt.plot(x_test_sorted, y_test_sorted_pred, label=f"Degree {degree}")
    
    y_train_pred = model.predict(x_train)
    train_errors.append(mean_squared_error(y_train, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

plt.title("Test predictions for different polynomial degrees (sorted)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# part(f)
plt.figure("Figure 3")
plt.plot(degrees, train_errors, label="Training Error", marker='o')
plt.plot(degrees, test_errors, label="Test Error", marker='o')
plt.xlabel("Polynomial Degree")
plt.ylabel("Mean Squared Error")
plt.title("Training and Test Errors vs Polynomial Degree")
plt.legend()
plt.show()