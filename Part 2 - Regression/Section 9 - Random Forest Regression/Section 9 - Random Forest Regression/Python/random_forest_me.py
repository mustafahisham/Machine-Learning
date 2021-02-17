# -*- coding: utf-8 -*-
"""
Created on Sun May  3 17:13:50 2020

@author: musta
"""


# -*- coding: utf-8 -*-
"""
Created on Sun May  3 15:51:06 2020

@author: musta
"""


#Random forest Regression

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
"""
# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""

#Fitting the regression model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 277, criterion = 'mse', random_state = 0)
regressor.fit(X,y)

#create your regressor here

#Predicting a new resulte with linear Regression
y_pred = regressor.predict(np.array(6.5).reshape(1,-1))

#Visualising the Polynomial regression set results for higher resolution
X_grid = np.arange(min(X), max(X), step = 0.01) #for Higher resresolution
X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("possition level")
plt.ylabel("Salary")
plt.show() 










































