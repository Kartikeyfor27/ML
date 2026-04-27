import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression

sns.set_style('whitegrid')

# Load sample Iris dataset
# Use IRIS.csv from workspace and inject missing values for demonstration
raw_df = pd.read_csv('IRIS.csv')
df = raw_df.copy()

np.random.seed(42)
missing_indices = np.random.choice(df.index, size=12, replace=False)
cols_to_mask = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for idx in missing_indices:
    col = np.random.choice(cols_to_mask)
    df.loc[idx, col] = np.nan

print('Loaded IRIS dataset and injected missing values for demonstration.')
print(f'Total missing values: {df.isnull().sum().sum()}')





print('Dataset snapshot with missing values:')
print(df.head(10))

print('\nMissing values count by column:')
print(df.isnull().sum())

plt.figure(figsize=(8, 4))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.show()





deleted_df = df.dropna()

print('After deletion, dataset shape: ', deleted_df.shape)
print('Missing values remaining:')
print(deleted_df.isnull().sum())
print('\nPreview of dataset after deletion:')
print(deleted_df.head())




#aapplying imputation methods
numeric_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

# 1. Mean Imputation
mean_imputed = df.copy()
mean_values = mean_imputed[numeric_cols].mean()
mean_imputed[numeric_cols] = mean_imputed[numeric_cols].fillna(mean_values)
print('Mean Imputation - first 5 rows:')
print(mean_imputed.head())
print('\nMissing values after mean imputation:')
print(mean_imputed.isnull().sum())


# 2. Median Imputation
median_imputed = df.copy()
median_values = median_imputed[numeric_cols].median()
median_imputed[numeric_cols] = median_imputed[numeric_cols].fillna(median_values)
print('Median Imputation - first 5 rows:')
print(median_imputed.head())
print('\nMissing values after median imputation:')
print(median_imputed.isnull().sum())


# 3. Mode Imputation
mode_imputed = df.copy()
for col in numeric_cols:
    mode_value = mode_imputed[col].mode()[0]
    mode_imputed[col] = mode_imputed[col].fillna(mode_value)
print('Mode Imputation - first 5 rows:')
print(mode_imputed.head())
print('\nMissing values after mode imputation:')
print(mode_imputed.isnull().sum())
