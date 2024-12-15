import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load the dataset
data_path = 'winequality-red.csv'  # Replace with your file path
column_names = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol", "quality"
]
# Load data
data = pd.read_csv(data_path, sep=';', names=column_names, skiprows=1, engine='python')
data.columns = data.columns.str.strip()
# Plot the bar graph for number of samples per quality level
plt.figure(figsize=(10, 6))
data['quality'].value_counts().sort_index().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Number of Samples for Each Quality Level', fontsize=14)
plt.xlabel('Quality', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
# Generate bar plots for all features in one figure
plt.figure(figsize=(20, 15))
for i, column in enumerate(data.columns):
    plt.subplot(4, 3, i + 1)  # Create a subplot grid for all features
    plt.hist(data[column], bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {column}')

plt.tight_layout()
plt.suptitle('Feature Distributions', fontsize=16, y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# PCA Visualization
features = data.iloc[:, :-1]  # Exclude the target column (quality)
pca = PCA(n_components=2)  # Reduce to 2 components for visualization
principal_components = pca.fit_transform(features)
pca_data = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_data['Quality'] = data['quality']  # Add quality as a label

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Quality', data=pca_data, palette='viridis', alpha=0.7)
plt.title('PCA Visualization of the Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Quality', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
