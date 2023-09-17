# -*- coding: utf-8 -*-
"""

@author: arsha
"""

# Importing the data  file

import pandas as pd
df = pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 13 KNN data files/zoo.csv")
df

df.shape         
df.info()        # Data type 
df.describe()    
df.isna().sum()  


# Exploratory Data Analysis (EDA)
# Correlation Analysis

df.corr()

import seaborn as sns 
sns.set_style(style='darkgrid')
sns.pairplot(df)

''' The given Data set is in binary format so we have to apply data transformation techniques'''


# Split the X and Y variable

Y=df['type']
Y
X=df.iloc[:,1:17]
X


# Data partation
# Train and Test
 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.70)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape


# Selecting few models
# Model fitting for KNeighborsClassifer

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X_train,Y_train)


# Using gridsearchCV for finding best n_neighbors

n_neighbors=list(range(1,50))
parameters={'n_neighbors':n_neighbors}
from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=KNN, param_grid=parameters)
grid.fit(X_train,Y_train)
print(grid.best_score_)
print(grid.best_params_)


# Model predictions

Y_pred_train = KNN.predict(X_train)
Y_pred_test = KNN.predict(X_test)


# Metrics

from sklearn.metrics import accuracy_score
acc1 = accuracy_score(Y_train,Y_pred_train)
print("Training Accuracy: ",acc1.round(2))
acc2 = accuracy_score(Y_test,Y_pred_test)
print("Test Accuracy: ",acc2.round(2))


# Cross validation techniques
# K-Fold validation

from sklearn.model_selection import KFold
Kf=KFold(5)
Training_mse=[]
Test_mse=[]
for train_index,test_index in Kf.split(X):
    X_train,X_test=X.loc[train_index],X.iloc[test_index]
    Y_train,Y_test=Y.iloc[train_index],Y.iloc[test_index]
    KNN.fit(X_train,Y_train)
    Y_train_pred=KNN.predict(X_train)
    Y_test_pred=KNN.predict(X_test)

Training_mse.append(accuracy_score(Y_train,Y_train_pred))
Test_mse.append(accuracy_score(Y_test,Y_test_pred))


import numpy as np
print('training accuracy score:',np.mean(Training_mse).round(3))    
print('test accuracy score:',np.mean(Test_mse).round(3)) 


# Final Model 

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X,Y)
df.info()


# Final Model predictions

dt=pd.DataFrame({'hair':0,'feathers':0,'eggs':1,'milk':0,'airborne':0,
                 'aquatic':1,'predator':1,'toothed':1,'backbone':1,'breathes':0,
                 'venomous':0,'fins':1,'legs':0,'tail':1,'domestic':0,'catsize':0},index=[0])
t1=KNN.predict(dt)
t1














































\