import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set_style('whitegrid')
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print('Training shape:', X_train.shape)
print('Testing shape:', X_test.shape)



kernels = ['linear', 'poly', 'rbf', 'sigmoid']
results = []
models = {}

for kernel in kernels:
    model = SVC(kernel=kernel, gamma='scale', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({'Kernel': kernel, 'Accuracy': acc})
    models[kernel] = {'model': model, 'y_pred': y_pred}

results_df = pd.DataFrame(results)
results_df


plt.figure(figsize=(8, 5))
sns.barplot(data=results_df, x='Kernel', y='Accuracy', palette='muted')
plt.title('SVM Accuracy by Kernel')
plt.ylim(0.8, 1.0)
plt.ylabel('Test Accuracy')
plt.show()
