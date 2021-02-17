# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 14:29:50 2020

@author: musta
"""


# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

"""
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
#Fitting linear regression to the dataset
from sklearn.linear_model import LinearRegression 
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualising the linear regression set results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title("Truth or Bluff (linear regression)")
plt.xlabel("possition level")
plt.ylabel("Salary")
plt.show() 


#Visualising the Polynomial regression set results
X_grid = np.arange(min(X), max(X) + 0.1,step = 0.1) #for Higher resresolution
X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)
), color = 'blue')
plt.title("Truth or Bluff (Polynomaial regression)")
plt.xlabel("possition level")
plt.ylabel("Salary")
plt.show() 

#Predicting a new resulte with linear Regression
lin_reg.predict(np.array(6.5).reshape(1,-1))

#Predicting a new resulte with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(np.array(6.5).reshape(1,-1)))










































