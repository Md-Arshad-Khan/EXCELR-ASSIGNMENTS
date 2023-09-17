# -*- coding: utf-8 -*-
"""

@author: arsha
"""

# Importing the data file

import pandas as pd
df=pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 14 Decision Trees data files/Company_Data.csv")

df.columns  
df['Sales'].max()     # Maximum Sales Value
df['Sales'].min()           # Minimum Sales Value
df['Sales'].median()            # Median Sales Value
df['Sales'].mean()                  # Mean Sales Value


''' For the given Data set sales is the target variable so we are taking the average
sales  value which is 7.49 from the average sales we are taking greater than 7.49
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


# Exploratory Data Analysis (EDA)

import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)


# Data Transformation
# Standard Scaling on Continiuous Variables

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


# Splitting the  Variables into X and Y

X=df.iloc[:,0:11]
X
Y=df['Target']
Y


# Data partition
# Train and Test

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)



# Model fitting  # DecisionTreeClassifier
# Criterion='entropy'

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='entropy')
DT.fit(X_train,Y_train)


# Model predictions

Y_pred_train = DT.predict(X_train)
Y_pred_test = DT.predict(X_test)


# Metrics

from sklearn.metrics import accuracy_score
acc1 = accuracy_score(Y_train, Y_pred_train)
print("Training Accuracy:", (acc1).round(2))
acc2 = accuracy_score(Y_test, Y_pred_test)
print("Test Accuracy:", (acc2).round(2))


# criterion='gini'

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini')
DT.fit(X_train,Y_train)


# Model predictions

Y_pred_train = DT.predict(X_train)
Y_pred_test = DT.predict(X_test)


# Metrics

from sklearn.metrics import accuracy_score
acc1 = accuracy_score(Y_train, Y_pred_train)
print("Training Accuracy:", (acc1).round(2))
acc2 = accuracy_score(Y_test, Y_pred_test)
print("Test Accuracy:", (acc2).round(2))

''' We have fitted the model with both gini and entropy criterion
    in both cases we got Train accuracy score as 100 and Test Accuracy as 99 '''
    
''' So the accuracy score is 100,hence we need not to perform 
   ensemble methods like bagging,randomforest,ada boost etc....'''


























