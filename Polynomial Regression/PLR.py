import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

from sklearn.linear_model import LinearRegression 
regressor_linear = LinearRegression()
regressor_linear.fit(X, y)

# Polynomial regression
from sklearn.preprocessing import PolynomialFeatures 
regressor_poly = PolynomialFeatures(degree = 4)
X_poly = regressor_poly.fit_transform(X)

regressor_linear2 = LinearRegression()
regressor_linear2.fit(X_poly, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, regressor_linear.predict(X), color = 'blue')
plt.title('Linear Regressor Prediction')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor_linear2.predict(regressor_poly.fit_transform(X_grid)), color = 'blue')
plt.title('Polynomial Regressor Prediction')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

candidate = regressor_poly.fit_transform(6.5)
y_pred_linear = regressor_linear.predict(6.5)

y_pred_poly = regressor_linear2.predict(candidate)

