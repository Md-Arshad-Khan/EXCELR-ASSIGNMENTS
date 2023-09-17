# -*- coding: utf-8 -*-
"""

@author: arsha
"""

# Importing the data file


import pandas as pd
df=pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 15 Random forest data files/Company_Data.csv")
df


df.columns  
df['Sales'].max()   # Maximum Sales Value
df['Sales'].min()      # Minimum Sales Value
df['Sales'].median()       # Median Sales Value
df['Sales'].mean()             # Mean Sales Value


''' For the given Data set sales is the target variable so we have taken the average
sales  value which is 7.49 from the average sales we are taking the sales greater than 7.49
as "High" sales and less  than 7.49 as "Low" '''


# Converting Sales into High and Low using "For Loop"

for index, row in df.iterrows():
    taxable_income = row['Sales']
    if taxable_income <= 7.49:
        classification = "Low"
    else:
        classification = "High"
    df.at[index, 'Target'] = classification
df
df.info()


#  Exploratory data Analysis (EDA)

import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)


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

X=df.iloc[:,0:11]
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
                  n_estimators=400,
                  max_samples=0.4,
                  max_features=0.1,
                  random_state=9)



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

d1={'n_estimators':[100,200,300,400],
    'max_samples':[0.2,0.4,0.6,0.8],
    'max_features':[0.1,0.3,0.5,0.7],
    'random_state':[3,5,7,9]}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=RFC, param_grid=d1)
grid.fit(X_train,Y_train)
print(grid.best_score_)
print(grid.best_params_)


















































