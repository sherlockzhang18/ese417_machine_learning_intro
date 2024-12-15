import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data_path = 'winequality-red.csv'  # Replace with your actual file path
column_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality"
]
data = pd.read_csv(data_path, sep=';', names=column_names, skiprows=1, engine='python')
data.columns = data.columns.str.strip()

# 1. Non-linear Transformation: Apply log transformation to skewed features
log_transform_cols = ['volatile acidity', 'chlorides', 'residual sugar', 'sulphates']
for col in log_transform_cols:
    data[col] = np.log1p(data[col])  # log1p ensures log(0) doesn't occur

# 2. Dimensionality Reduction: Apply PCA
pca = PCA(n_components=5)  # Reduce to 5 principal components
X = data.iloc[:, :-1]  # All features except 'quality'
X_pca = pca.fit_transform(X)

# Create a DataFrame for PCA results
pca_columns = [f'PCA_Component_{i+1}' for i in range(X_pca.shape[1])]
pca_df = pd.DataFrame(X_pca, columns=pca_columns)

# 3. Combine PCA results with the target variable
data_pca = pd.concat([pca_df, data[['quality']]], axis=1)

# 4. Normalization: Scale features to [0, 1]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_pca.iloc[:, :-1])  # Exclude 'quality'
scaled_df = pd.DataFrame(scaled_data, columns=pca_columns)

# Combine scaled data with the target variable
final_data = pd.concat([scaled_df, data[['quality']]], axis=1)

# Save the processed data to a new CSV file
output_path = 'processed_winequality_data.csv'
final_data.to_csv(output_path, index=False)

print(f"Processed data saved to {output_path}")
