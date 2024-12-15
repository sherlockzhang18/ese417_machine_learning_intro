import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the processed dataset
processed_data_path = 'processed_winequality_data.csv'  # Replace with your actual processed file path
processed_data = pd.read_csv(processed_data_path)

# Timing the process for unmodified and resampled datasets
start_unmodified = time.time()

# Split features and target
X = processed_data.iloc[:, :-1]  # Exclude 'quality'
y = processed_data['quality']   # Target variable

# Split unmodified data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Hyperparameter Optimization with GridSearchCV (Unmodified Data)
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

# Evaluate ANN on unmodified dataset
best_mlp_unmodified = grid_search.best_estimator_
best_mlp_unmodified.fit(X_train, y_train)
y_pred_unmodified = best_mlp_unmodified.predict(X_test)
accuracy_unmodified = accuracy_score(y_test, y_pred_unmodified)
print("\nPerformance on Unmodified Dataset:")
print("Accuracy (Unmodified):", accuracy_unmodified)
print(classification_report(y_test, y_pred_unmodified))

end_unmodified = time.time()
print(f"\nTime taken for unmodified data: {end_unmodified - start_unmodified:.2f} seconds")

# Timing the process for resampled data
start_resampled = time.time()

# Step 2: Balance the Dataset Using Resampling
# Combine features and target for resampling
data_combined = pd.concat([X, y], axis=1)
majority_class = data_combined[data_combined['quality'] == y.value_counts().idxmax()]
minority_classes = data_combined[data_combined['quality'] != y.value_counts().idxmax()]

# Upsample minority classes
from sklearn.utils import resample
upsampled_minority = resample(minority_classes, 
                               replace=True, 
                               n_samples=len(majority_class), 
                               random_state=42)

balanced_data = pd.concat([majority_class, upsampled_minority])
X_balanced = balanced_data.iloc[:, :-1]
y_balanced = balanced_data.iloc[:, -1]

# Split resampled data into train and test sets
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42)

# Hyperparameter Optimization with GridSearchCV (Resampled Data)
grid_search_resampled = GridSearchCV(mlp, param_grid, cv=3, scoring='accuracy', n_jobs=1)
grid_search_resampled.fit(X_train_balanced, y_train_balanced)

# Evaluate ANN on resampled dataset
best_mlp_resampled = grid_search_resampled.best_estimator_
best_mlp_resampled.fit(X_train_balanced, y_train_balanced)
y_pred_resampled = best_mlp_resampled.predict(X_test_balanced)
accuracy_resampled = accuracy_score(y_test_balanced, y_pred_resampled)
print("\nPerformance on Resampled Dataset:")
print("Accuracy (Resampled):", accuracy_resampled)
print(classification_report(y_test_balanced, y_pred_resampled))

end_resampled = time.time()
print(f"\nTime taken for resampled data: {end_resampled - start_resampled:.2f} seconds")

# Accuracy Comparison
print("\nAccuracy Comparison:")
print(f"Unmodified Dataset Accuracy: {accuracy_unmodified:.2f}")
print(f"Resampled Dataset Accuracy: {accuracy_resampled:.2f}")
