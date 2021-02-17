# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#splitting data into training and test set
from sklearn.model_selection import train_test_split
X_train , X_test , y_train, y_test = train_test_split(X, y ,test_size = 0.2, random_state = 0)

# Feature scaling
""""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting simple regression to the training set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)

#Visualising the training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, y_pred_train, color = 'blue')
plt.title("salary vs Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show() 


#Visualising the test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, y_pred_train, color = 'blue')
plt.title("salary vs Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show() 
