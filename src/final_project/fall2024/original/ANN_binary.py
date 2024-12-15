import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data_path = 'winequality-red.csv'  # Replace with your file path
column_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality"
]
data = pd.read_csv(data_path, sep=';', names=column_names, skiprows=1, engine='python')
data.columns = data.columns.str.strip()

# Convert the problem to binary classification
# Low quality: quality <= 5, High quality: quality >= 6
data['quality'] = data['quality'].apply(lambda x: 0 if x <= 5 else 1)

# Split features and target
X = data.iloc[:, :-1]
y = data['quality']

# Step 1: Balance Dataset Using Resampling
data_combined = pd.concat([X, y], axis=1)
majority_class = data_combined[data_combined['quality'] == y.value_counts().idxmax()]
minority_class = data_combined[data_combined['quality'] != y.value_counts().idxmax()]

upsampled_minority = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
balanced_data = pd.concat([majority_class, upsampled_minority])

X_balanced = balanced_data.iloc[:, :-1]
y_balanced = balanced_data.iloc[:, -1]

# Normalize the features
scaler = MinMaxScaler()
X_balanced_scaled = scaler.fit_transform(X_balanced)
X_scaled = scaler.transform(X)  # Keep this for unbalanced comparison

# Split datasets
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(
    X_balanced_scaled, y_balanced, test_size=0.2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define Hyperparameters
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive']
}

# Step 2: GridSearchCV for Binary Classification
mlp = MLPClassifier(max_iter=1000, random_state=42, early_stopping=True)
grid_search_balanced = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy', n_jobs=1)
grid_search_balanced.fit(X_train_balanced, y_train_balanced)

# Best model on balanced data
best_mlp_balanced = grid_search_balanced.best_estimator_
print("Best Hyperparameters (Balanced Binary):", grid_search_balanced.best_params_)

# Train and Evaluate on Balanced Data
best_mlp_balanced.fit(X_train_balanced, y_train_balanced)
y_pred_balanced = best_mlp_balanced.predict(X_test_balanced)
print("\nPerformance on Balanced Dataset (Binary):")
print("Accuracy:", accuracy_score(y_test_balanced, y_pred_balanced))
print(classification_report(y_test_balanced, y_pred_balanced))

# Step 3: Comparison with Unbalanced Dataset
grid_search_unbalanced = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy', n_jobs=1)
grid_search_unbalanced.fit(X_train, y_train)

# Best model on unbalanced data
best_mlp_unbalanced = grid_search_unbalanced.best_estimator_
print("Best Hyperparameters (Unbalanced Binary):", grid_search_unbalanced.best_params_)

# Train and Evaluate on Unbalanced Data
best_mlp_unbalanced.fit(X_train, y_train)
y_pred_unbalanced = best_mlp_unbalanced.predict(X_test)
print("\nPerformance on Unbalanced Dataset (Binary):")
print("Accuracy:", accuracy_score(y_test, y_pred_unbalanced))
print(classification_report(y_test, y_pred_unbalanced))

# Comparison Summary
print("\nSummary of Results (Binary Classification):")
print(f"Accuracy (Balanced Binary Dataset): {accuracy_score(y_test_balanced, y_pred_balanced):.2f}")
print(f"Accuracy (Unbalanced Binary Dataset): {accuracy_score(y_test, y_pred_unbalanced):.2f}")
