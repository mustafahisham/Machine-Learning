# Data Preprocessing Template

# Importing the libraries
""""import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

code above
////////////////////////////////////////
# Feature Scaling
  from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset 
dataset = pd.read_csv('Data.csv')
#declare the in and dependant variables
X = dataset.iloc[: , :-1]
y = dataset.iloc[: ,3]

# complete missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean',verbose=0)
imputer = imputer.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])

#Encoding the categorical data (covert the strings into dummy variable)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    transformers = [('one_hot_encoder', OneHotEncoder(), [0])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

"""
columnTransformer = ColumnTransformer([('encoder', 
                                        OneHotEncoder(), 
                                        [1, 2, 3])], 
                                      remainder='passthrough') 
  
X = np.array(columnTransformer.fit_transform(X), dtype = np.float) 
"""
X = np.array(ct.fit_transform(X), dtype=np.float)

# covert the strings in independent column into dummy value 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#splitting data into training and test set
from sklearn.model_selection import train_test_split
X_train , X_test , y_train, y_test = train_test_split(X, y ,test_size = 0.2, random_state = 0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)










