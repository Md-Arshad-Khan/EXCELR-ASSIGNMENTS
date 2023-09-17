# -*- coding: utf-8 -*-
"""

@author: arsha
"""


# Importing the data file

import pandas as pd
df=pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 15 Random forest data files/Fraud_check.csv")
df

df.shape
df.columns 
df.info() 


''' Here the target variable is given indirectly which is taxable income, 
    so taxable_income<=30000 is consideres as risk and taxable_income>30000 
   is considered as good.for that we are performing the iterations through the 
   rows of taxable_income and creating the target variable using For Loop'''

for index, row in df.iterrows():
    taxable_income = row['Taxable.Income']
    if taxable_income <= 30000:
        classification = "risk"
    else:
        classification = "good"
    df.at[index, 'Target'] = classification
df


df['Target'].value_counts()         # Value counts of Target
df['Undergrad'].value_counts()       
df['Marital.Status'].value_counts() 
df['Urban'].value_counts() 


# Exploratory Data Analysis (EDA)

import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)


# Countplot of all discrete variables

sns.countplot(x='Target',data=df)
sns.countplot(x='Undergrad',data=df)
sns.countplot(x='Marital.Status',data=df)
sns.countplot(x='Urban',data=df)


# Data Transformations
# Standard Scaling on Continious Variables

from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
for column in df.columns:
    if df[column].dtype == object:
        continue
    df[column]=SS.fit_transform(df[[column]])
df


# Labelencoding on discrete variables

import numpy as np

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
for column in df.columns:
    if df[column].dtype == np.number:
        continue
    df[column]=LE.fit_transform(df[column])
df
df.info()


# Splitting the Varibles into X and Y 

X=df.iloc[:,0:6]
X
Y=df['Target']
Y


# Data partition
# Train and Test

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)


# Random Forest 

from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(
                  n_estimators=150,
                  max_samples=0.9,
                  max_features=0.3,
                  random_state=6)


# Model Fitting

RFC.fit(X_train,Y_train)
Y_pred_train = RFC.predict(X_train)
Y_pred_test = RFC.predict(X_test)


# Metrics

from sklearn.metrics import accuracy_score
acc1 = accuracy_score(Y_train, Y_pred_train)
print("Training Accuracy:", (acc1).round(2))
acc2 = accuracy_score(Y_test, Y_pred_test)
print("Test Accuracy:", (acc2).round(2))


# Grid Search CV

d1={'n_estimators':[50,150,200,250],
    'max_samples':[0.1,0.5,0.7,0.9],
    'max_features':[0.3,0.5,0.7,0.9],
    'random_state':[2,6,8,10]}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=RFC, param_grid=d1)
grid.fit(X_train,Y_train)
print(grid.best_score_)
print(grid.best_params_)



















































