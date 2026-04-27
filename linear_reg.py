import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

plt.style.use('seaborn-v0_8')

salary_df = pd.read_csv('Salary_dataset.csv')
salary_df.head()


X_simple = salary_df[['YearsExperience']]
y = salary_df['Salary']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

simple_model = LinearRegression()
simple_model.fit(X_train_s, y_train_s)
y_pred_s = simple_model.predict(X_test_s)

print('Simple Regression coefficients:', simple_model.coef_)
print('Simple Regression intercept:', simple_model.intercept_)
print('MSE:', mean_squared_error(y_test_s, y_pred_s))
print('R² score:', r2_score(y_test_s, y_pred_s))


plt.figure(figsize=(8, 5))
plt.scatter(X_test_s, y_test_s, color='blue', label='Actual')
plt.plot(X_test_s, y_pred_s, color='red', linewidth=2, label='Predicted')
plt.title('Simple Linear Regression: Salary vs YearsExperience')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.legend()
plt.show()