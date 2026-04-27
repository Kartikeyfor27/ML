# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('IRIS.csv')



# Create box plots to visualize outliers
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Box Plots - Visualizing Outliers', fontsize=14, fontweight='bold')

numeric_cols = df.select_dtypes(include=[np.number]).columns
for idx, col in enumerate(numeric_cols):
    ax = axes[idx // 2, idx % 2]
    ax.boxplot(df[col], vert=True)
    ax.set_title(f'Box Plot: {col}')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Display quartiles and IQR information
print("Quartile and IQR Information:")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    print(f"\n{col}:")
    print(f"  Q1 (25%): {Q1:.4f}")
    print(f"  Q3 (75%): {Q3:.4f}")
    print(f"  IQR: {IQR:.4f}")




# Function to remove outliers using IQR method
def remove_outliers_iqr(data, columns):
    """Remove outliers using the IQR method"""
    df_clean = data.copy()
    outliers_removed = 0
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers before removing
        outliers_in_col = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        outliers_removed += outliers_in_col
        
        # Remove outliers
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        
        print(f"{col}:")
        print(f"  Lower Bound: {lower_bound:.4f}")
        print(f"  Upper Bound: {upper_bound:.4f}")
        print(f"  Outliers found: {outliers_in_col}")
    
    return df_clean, outliers_removed

# Apply IQR method to remove outliers
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_clean, total_outliers = remove_outliers_iqr(df, numeric_cols)

print("\n" + "="*50)
print(f"Original dataset shape: {df.shape}")
print(f"Dataset shape after outlier removal: {df_clean.shape}")
print(f"Total rows removed: {df.shape[0] - df_clean.shape[0]}")



# Compute correlation matrix
correlation_matrix = df_clean.corr(numeric_only=True)

print("Correlation Matrix:")
print(correlation_matrix)

# Create a heatmap visualization
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Display correlations with target variable (species)
print("Feature Correlations (Numeric features only):")
print(correlation_matrix.iloc[:, :-1])