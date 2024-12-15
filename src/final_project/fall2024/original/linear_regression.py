import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Load the dataset
data_path = 'winequality-red.csv'
column_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality"
]
data = pd.read_csv(data_path, sep=';', names=column_names, skiprows=1, engine='python')
data.columns = data.columns.str.strip()

# Check data
print("Fixed Data:")
print(data.head())
print(data.info())

# Data Splitting
X = data.iloc[:, :-1].values  # Input features (11 physicochemical attributes)
y = data.iloc[:, -1].values   # Output target (quality score)

# Train Linear Regression model using sklearn
model = LinearRegression()
model.fit(X, y)

# Output model coefficients and intercept in one row
coefficients = [f"{name}: {coef:.3f}" for name, coef in zip(column_names[:-1], model.coef_)]
print(f"Intercept: {model.intercept_:.3f}, " + ", ".join(coefficients))

# Calculate p-values manually
n = len(y)  # Number of observations
p = X.shape[1]  # Number of predictors
predictions = model.predict(X)
residuals = y - predictions
rss = np.sum(residuals**2)  # Residual Sum of Squares
sigma_squared = rss / (n - p - 1)  # Variance of residuals
X_with_const = np.hstack([np.ones((X.shape[0], 1)), X])  # Add constant for intercept
cov_matrix = np.linalg.inv(X_with_const.T @ X_with_const) * sigma_squared
standard_errors = np.sqrt(np.diag(cov_matrix))

# Calculate t-statistics and p-values
t_stats = model.coef_ / standard_errors[1:]  # Exclude intercept's standard error
p_values = [2 * (1 - stats.t.cdf(np.abs(t), df=n - p - 1)) for t in t_stats]

# Output p-values with feature names in one row
p_value_output = [f"{name}: {p_value}" for name, p_value in zip(column_names[:-1], p_values)]
print("P-values: " + ", ".join(p_value_output))

# Evaluate model performance
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)