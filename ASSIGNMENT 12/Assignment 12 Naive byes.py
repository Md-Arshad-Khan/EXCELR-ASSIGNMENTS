# -*- coding: utf-8 -*-
"""

@author: arsha
"""


# Importing the data files

import pandas as pd

df_train=pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 12 Naive byes data files/SalaryData_Train.csv")
df_train

df_test=pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 12 Naive byes data files/SalaryData_Test.csv")
df_test

df_train.shape
df_test.shape


#    Train Dataset

df_train.info()         
df_train.describe()     
df_train.isna().sum()
df_train['Salary'].value_counts()


# Drop the duplicates

train=df_train.drop_duplicates()
train.head()


# Exploratory Data Analysis (EDA) for train dataset

import seaborn as sns 
sns.set_style(style='darkgrid')
sns.pairplot(train)


# Countplot for Target variable

sns.countplot(x='Salary',data=train)


# Data transformations on Train dataset
# Labelencoding

import numpy as np

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for column in train.columns:
    if train[column].dtype == np.number:
        continue
    train[column]=LE.fit_transform(train[column])

train.info()


# Test Dataset

df_test.info()
df_test.describe()
df_test.isna().sum()
df_test['Salary'].value_counts()


# Drop the duplicates
 
test=df_test.drop_duplicates()
test.head()


# Exploratory Data Analysis (EDA) for test dataset

import seaborn as sns 
sns.set_style(style='darkgrid') 
sns.pairplot(test)


# Countplot for Target variable

sns.countplot(x='Salary',data=test)


# Data transformations on Test dataset
# Labelencoding

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for column in test.columns:
    if test[column].dtype == np.number:
        continue
    test[column]=LE.fit_transform(test[column])

test.info()


# X and Y Train variables of Train dataset

X_train=train.iloc[:,0:13]
X_train
Y_train=train['Salary']
Y_train


# X and Y Test variables of Test dataset

X_test=test.iloc[:,0:13]
X_test
Y_test=test['Salary']
Y_test


# Model fitting
# Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
MNB=MultinomialNB()
MNB.fit(X_train,Y_train)


# Model Predictions

Y_pred_train=MNB.predict(X_train)
Y_pred_test=MNB.predict(X_test)


# Metrics

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train, Y_pred_train)
print("Training Accuracy score:", (ac1*100).round(2))
ac2 = accuracy_score(Y_test, Y_pred_test)
print("Test Accuracy score:", (ac2*100).round(2))













































