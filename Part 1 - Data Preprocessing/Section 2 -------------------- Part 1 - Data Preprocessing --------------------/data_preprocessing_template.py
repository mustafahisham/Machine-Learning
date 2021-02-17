import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset 
dataset = pd.read_csv('Data.csv')
#declare the in and dependant variables
X = dataset.iloc[: , :-1]
y = dataset.iloc[: ,3]

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





