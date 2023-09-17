# -*- coding: utf-8 -*-
"""

@author: arsha
"""

# Importing the data file

import pandas as pd
df=pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 13 KNN data files/glass.csv")
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


# Splitting Variables into X and Y 

Y=df['Type']
Y
X=df.iloc[:,0:9]
X


# Data Transformations 

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss_x=ss.fit_transform(X)
ss_x=pd.DataFrame(ss_x)
ss_x.columns=['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']
ss_x


# Data partition
#   Training and Testing
 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.70)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape


# Selecting few models
# Model fitting for KNeighborsClassifer

from sklearn.neighbors import KNeighborsClassifier
KNN=KNeighborsClassifier()
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

Y_train_pred=KNN.predict(X_train)
Y_test_pred=KNN.predict(X_test)


# Metrics

from sklearn.metrics import accuracy_score
acc1 = accuracy_score(Y_train,Y_train_pred)
print("Training Accuracy: ",acc1)
acc2 = accuracy_score(Y_test,Y_test_pred)
print("Test Accuracy: ",acc2)


# Cross validation for all chosen models
# K-Fold validation

from sklearn.model_selection import KFold
Kf=KFold(10)
Training_mse=[]
Test_mse=[]
for train_index,test_index in Kf.split(X):
    X_train,X_test=X.iloc[train_index],X.iloc[test_index]
    Y_train,Y_test=Y.iloc[train_index],Y.iloc[test_index]
    KNN.fit(X_train,Y_train)
    Y_train_pred=KNN.predict(X_train)
    Y_test_pred=KNN.predict(X_test)


Training_mse.append(accuracy_score(Y_train,Y_train_pred))
Test_mse.append(accuracy_score(Y_test,Y_test_pred))


import numpy as np
print('trining accuracy score:',np.mean(Training_mse))  
print('test accuracy score:',np.mean(Test_mse))

''' so we cannot tune beyond this accuracy score,
S we will fit the model'''


# Final Model 

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X,Y)
df.info()
df


# Final Model predictions

dt=pd.DataFrame({'RI':1.51711,'Na':14.23,'Mg':0.00,'Al':2.08,'Si':73.36,
                 'K':0.00,'Ca':8.62,'Ba':1.67,'Fe':0.0},index=[0])
t1=KNN.predict(dt)
t1























































