import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import resample

# Load the original dataset
data_path = 'winequality-red.csv'  # Replace with your actual file path
column_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality"
]
data = pd.read_csv(data_path, sep=';', names=column_names, skiprows=1, engine='python')
data.columns = data.columns.str.strip()

# Timing the process for original data
start_original = time.time()

# Split features and target
X = data.iloc[:, :-1]  # Exclude 'quality'
y = data.iloc[:, -1]   # Target variable

# Step 1: Balance the Dataset Using Resampling
# Combine features and target for resampling
data_combined = pd.concat([X, y], axis=1)
majority_class = data_combined[data_combined['quality'] == y.value_counts().idxmax()]
minority_classes = data_combined[data_combined['quality'] != y.value_counts().idxmax()]

# Upsample minority classes
upsampled_minority = resample(minority_classes, 
                               replace=True, 
                               n_samples=len(majority_class), 
                               random_state=42)

balanced_data = pd.concat([majority_class, upsampled_minority])
X_balanced = balanced_data.iloc[:, :-1]
y_balanced = balanced_data.iloc[:, -1]

# Split balanced data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Step 2: Hyperparameter Optimization with GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive']
}
mlp = MLPClassifier(max_iter=500, random_state=42)
grid_search = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy', n_jobs=1)
grid_search.fit(X_train, y_train)

# Best hyperparameters and corresponding model
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)
best_mlp = grid_search.best_estimator_

# Step 3: Train and Evaluate ANN on the Balanced Dataset
print("\nPerformance without Bagging:")
best_mlp.fit(X_train, y_train)
y_pred_no_bagging = best_mlp.predict(X_test)
print("Accuracy (No Bagging):", accuracy_score(y_test, y_pred_no_bagging))
print(classification_report(y_test, y_pred_no_bagging))

# Step 4: Bagging Ensemble with MLP
bagging_mlp = BaggingClassifier(base_estimator=best_mlp, n_estimators=10, random_state=42)
bagging_mlp.fit(X_train, y_train)
y_pred_bagging = bagging_mlp.predict(X_test)

print("\nPerformance with Bagging:")
print("Accuracy (With Bagging):", accuracy_score(y_test, y_pred_bagging))
print(classification_report(y_test, y_pred_bagging))

end_original = time.time()
print(f"\nTime taken for original data: {end_original - start_original:.2f} seconds")
