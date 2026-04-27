import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the Iris dataset
df = pd.read_csv('IRIS.csv')

# Display detailed information about the dataset


print("\nDataFrame Info:")
print(df.info())


print("STATISTICAL SUMMARY")
print(df.describe())


print("DATA TYPES")
print(df.dtypes)

print("DATASET COLUMNS")
print(f"Column Names: {df.columns.tolist()}")
print(f"Number of Columns: {len(df.columns)}")
print(f"Number of Rows: {len(df)}")
print(f"Duplicate rows: {df.duplicated().sum()}")





# Check for missing values
print("MISSING VALUES ANALYSIS")

missing_values = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

print("\nMissing Values Count:")
print(missing_values)

print("\nMissing Values Percentage:")
print(missing_percentage)

print("\nTotal Missing Values in Dataset:", df.isnull().sum().sum())

# Visualization of missing values
if missing_values.sum() > 0:
    plt.figure(figsize=(10, 4))
    sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
    plt.title('Missing Values Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("\n✓ No missing values found in the dataset!")

# Check for duplicate values

print("DUPLICATE VALUES ANALYSIS")


total_duplicates = df.duplicated().sum()
print(f"\nTotal Duplicate Rows: {total_duplicates}")

print(f"Percentage of Duplicates: {(total_duplicates/len(df))*100:.2f}%")

if total_duplicates > 0:
    print("\nDuplicate Rows:")
    print(df[df.duplicated(keep=False)].sort_values(by=df.columns.tolist()))
else:
    print("\n✓ No duplicate rows found in the dataset!")

# Check duplicates by specific columns

print("DUPLICATES BY FEATURE COMBINATIONS")


numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
duplicates_by_features = df[numeric_cols].duplicated().sum()
print(f"Duplicates (considering only numeric features): {duplicates_by_features}")
print(f"Percentage: {(duplicates_by_features/len(df))*100:.2f}%")    

