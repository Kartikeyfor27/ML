import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)


df = pd.read_csv('IRIS.csv')
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

#preparing Data 
scaler = StandardScaler()
X = scaler.fit_transform(df[numeric_cols])

print('Scaled data shape:', X.shape)
print('Mean of scaled features (approx.):', np.round(X.mean(axis=0), 3))
print('Std of scaled features (approx.):', np.round(X.std(axis=0), 3))




max_clusters = 8
inertia = []
silhouette_scores = []

for k in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

# Best K using silhouette
best_k_silhouette = np.argmax(silhouette_scores) + 2

# Elbow detection (simple method: largest drop in inertia)
inertia_diff = np.diff(inertia)
best_k_elbow = np.argmin(inertia_diff) + 2

print("Best K (Silhouette):", best_k_silhouette)
print("Best K (Elbow approx):", best_k_elbow)

# Final decision logic
if best_k_silhouette == best_k_elbow:
    final_k = best_k_silhouette
else:
    # Prefer silhouette, but keep elbow in mind
    final_k = best_k_silhouette

print("Final chosen K:", final_k)




optimal_k = final_k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

df['cluster'] = cluster_labels
print('Cluster counts:')
print(df['cluster'].value_counts())

print('\nCluster centers (scaled feature space):')
print(np.round(kmeans.cluster_centers_, 3))



plt.figure(figsize=(8, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='tab10', s=80, edgecolor='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='X', label='Centers')
plt.xlabel('Scaled ' + numeric_cols[0])
plt.ylabel('Scaled ' + numeric_cols[1])
plt.title(f'K-means clustering with K={optimal_k}')
plt.legend(*scatter.legend_elements(), title='Cluster')
plt.show()

sns.countplot(x='cluster', data=df, palette='tab10')
plt.title('Cluster distribution')
plt.show()
