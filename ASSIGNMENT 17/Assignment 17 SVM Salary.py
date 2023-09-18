# -*- coding: utf-8 -*-
"""

@author: arsha
"""

# Importing the data file

import pandas as pd

df_train=pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 17 SVM data files/SalaryData_Train(1).csv")
df_train


df_test=pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 17 SVM data files/SalaryData_Test(1).csv")
df_test


df_train.shape
df_test.shape


# Train Dataset

df_train.info()         
df_train.describe()     
df_train.isna().sum()
df_train['Salary'].value_counts()


# Drop the duplicates

train=df_train.drop_duplicates()
train.head()


# Exploratory Data Analysis (EDA) on train data set

import seaborn as sns 
sns.set_style(style='darkgrid')
sns.pairplot(train)


# Countplot for discrete Variables

sns.countplot(x='Salary',data=train)
sns.countplot(x='workclass',data=train)
sns.countplot(x='education',data=train)
sns.countplot(x='maritalstatus',data=train)
sns.countplot(x='occupation',data=train)
sns.countplot(x='relationship',data=train)
sns.countplot(x='race',data=train)
sns.countplot(x='sex',data=train)
sns.countplot(x='native',data=train)


# Data transformations on Train dataset
#  Labelencoding

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


# Exploratory Data Analysis (EDA) on test dataset 

import seaborn as sns 
sns.set_style(style='darkgrid') 
sns.pairplot(test)


# Countplot for Target variable

sns.countplot(x='Salary',data=test)
sns.countplot(x='workclass',data=train)
sns.countplot(x='education',data=train)
sns.countplot(x='maritalstatus',data=train)
sns.countplot(x='occupation',data=train)
sns.countplot(x='relationship',data=train)
sns.countplot(x='race',data=train)
sns.countplot(x='sex',data=train)
sns.countplot(x='native',data=train)



# Data transformations on Test dataset
# Labelencoding

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for column in test.columns:
    if test[column].dtype == np.number:
        continue
    test[column]=LE.fit_transform(test[column])

test.info()



# Splitting up the X and Y variables for train dataset

X_train=train.iloc[:,0:13]
X_train
Y_train=train['Salary']
Y_train


# Splitting up the X and Y variables of Test dataset

X_test=test.iloc[:,0:13]
X_test
Y_test=test['Salary']
Y_test


# Model fitting

''' For support vector classifier we have three types of kernals which are 
linear,poly,rbf .we will fit 3 svm model based on their kernals'''


# Support Vector classifier 
# Kernal='linear'

from sklearn.svm import SVC
clf = SVC(kernel='linear',C=5.0)
clf.fit(X_train, Y_train)


# Y predictions

Y_pred_train = clf.predict(X_train)
Y_pred_test  = clf.predict(X_test)


# Metrics

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train, Y_pred_train)
print("Training Accuracy score:", (ac1*100).round(2))
ac2 = accuracy_score(Y_test, Y_pred_test)
print("Test Accuracy score:", (ac2*100).round(2))
print('Variaance between test and train accuracy',(ac1-ac2).round(2))


# Support Vector classifier
# Kernal='Poly

from sklearn.svm import SVC
clf = SVC(kernel='poly',degree=5)
clf.fit(X_train, Y_train)


# Y prediictions

Y_pred_train = clf.predict(X_train)
Y_pred_test  = clf.predict(X_test)


# Metrics

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train, Y_pred_train)
print("Training Accuracy score:", (ac1*100).round(2))
ac2 = accuracy_score(Y_test, Y_pred_test)
print("Test Accuracy score:", (ac2*100).round(2))
print('Variaance between test and train accuracy',(ac1-ac2).round(2))


# Support Vector classifier
# Kernal=rbf

from sklearn.svm import SVC
clf = SVC(kernel='rbf',gamma='scale')
clf.fit(X_train, Y_train)


# Y predictions

Y_pred_train = clf.predict(X_train)
Y_pred_test  = clf.predict(X_test)


# Metrics

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train, Y_pred_train)
print("Training Accuracy score:", (ac1*100).round(2))
ac2 = accuracy_score(Y_test, Y_pred_test)
print("Test Accuracy score:", (ac2*100).round(2))
print('Variaance between test and train accuracy',(ac1-ac2).round(2))


''' The Poly kernal has the best accuracy score when compared to linear and
 gama kernal ,so we will finalize the poly kernal '''


































































