import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sns.set_style('whitegrid')

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print('Mean after scaling (approx.):', np.round(X_scaled.mean(axis=0), 3))
print('Std after scaling:', np.round(X_scaled.std(axis=0), 3))


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print('Reduced feature shape:', X_pca.shape)
print('Explained variance ratio:', np.round(pca.explained_variance_ratio_, 3))
print('Total explained variance:', np.round(pca.explained_variance_ratio_.sum(), 3))


pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['species'] = y.map(dict(enumerate(iris.target_names)))

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='species', palette='deep', s=100)
plt.title('PCA Projection of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Species')
plt.show()


variance_df = pd.DataFrame({
    'Principal Component': ['PC1', 'PC2'],
    'Explained Variance Ratio': pca.explained_variance_ratio_
})
variance_df['Cumulative Variance'] = variance_df['Explained Variance Ratio'].cumsum()
variance_df