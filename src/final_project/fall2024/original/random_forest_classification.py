import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils import resample

# Load the dataset
data_path = 'winequality-red.csv'  # Replace with your file path
column_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality"
]
data = pd.read_csv(data_path, sep=';', names=column_names, skiprows=1, engine='python')

# Redefine quality as binary classification
data['quality'] = (data['quality'] > 5).astype(int)  # Class 1: quality > 5, Class 0: quality <= 5
X = data.iloc[:, :-1]
y = data['quality']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Base Model
base_model = RandomForestClassifier(random_state=42)
base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_test)
print("Binary Classification Base Model Accuracy:", accuracy_score(y_test, y_pred_base))

# 2. Normalization with MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
scaled_model = RandomForestClassifier(random_state=42)
scaled_model.fit(X_train_scaled, y_train)
y_pred_scaled = scaled_model.predict(X_test_scaled)
print("Normalized Binary Model Accuracy:", accuracy_score(y_test, y_pred_scaled))

# 3. Standardization with StandardScaler
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)
std_model = RandomForestClassifier(random_state=42)
std_model.fit(X_train_std, y_train)
y_pred_std = std_model.predict(X_test_std)
print("Standardized Binary Model Accuracy:", accuracy_score(y_test, y_pred_std))

# 4. Dimensionality Reduction with PCA
pca = PCA(n_components=8)  # Reduce to 8 principal components
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
pca_model = RandomForestClassifier(random_state=42)
pca_model.fit(X_train_pca, y_train)
y_pred_pca = pca_model.predict(X_test_pca)
print("PCA Binary Model Accuracy:", accuracy_score(y_test, y_pred_pca))

# 5. Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy', n_jobs=1)
grid_search.fit(X_train_std, y_train)

best_rf_model = grid_search.best_estimator_
print("Best Hyperparameters (Binary):", grid_search.best_params_)

# Evaluate Best Model
y_pred_best = best_rf_model.predict(X_test_std)
print("GridSearch Optimized Binary Model Accuracy:", accuracy_score(y_test, y_pred_best))

# 6. Balancing Dataset Using Resampling
data_combined = pd.concat([X, y], axis=1)
majority_class = data_combined[data_combined['quality'] == y.value_counts().idxmax()]
minority_class = data_combined[data_combined['quality'] != y.value_counts().idxmax()]
upsampled_minority = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
balanced_data = pd.concat([majority_class, upsampled_minority])

X_balanced = balanced_data.iloc[:, :-1]
y_balanced = balanced_data.iloc[:, -1]
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)

X_train_balanced_std = std_scaler.fit_transform(X_train_balanced)
X_test_balanced_std = std_scaler.transform(X_test_balanced)

balanced_model = RandomForestClassifier(random_state=42)
balanced_model.fit(X_train_balanced_std, y_train_balanced)
y_pred_balanced = balanced_model.predict(X_test_balanced_std)
print("Balanced Binary Model Accuracy:", accuracy_score(y_test_balanced, y_pred_balanced))

# Summary of Results
print("\nSummary of Binary Classification Accuracies:")
print(f"Base Model: {accuracy_score(y_test, y_pred_base):.2f}")
print(f"Normalized Model: {accuracy_score(y_test, y_pred_scaled):.2f}")
print(f"Standardized Model: {accuracy_score(y_test, y_pred_std):.2f}")
print(f"PCA Model: {accuracy_score(y_test, y_pred_pca):.2f}")
print(f"GridSearch Optimized Model: {accuracy_score(y_test, y_pred_best):.2f}")
print(f"Balanced Model: {accuracy_score(y_test_balanced, y_pred_balanced):.2f}")