# -*- coding: utf-8 -*-
"""

@author: arsha
"""

# Importing the data file

import pandas as pd
df = pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 6 Logistic Regression data files/bank-full.csv",sep = ";")
df
df.info()



#   Data transformation
# Standard scaling

from sklearn.preprocessing import StandardScaler
SS = StandardScaler()

for column in df.columns:
    if df[column].dtype=='object':
        continue
    df[[column]]=SS.fit_transform(df[[column]])
df.info()    


# Labelencoding

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

import numpy as np 

for column in df.columns:
    if df[column].dtype == np.number:
        continue
    df[column]=LE.fit_transform(df[column])
df.info()
df


#  Splitting the variables

Y=df['y']
Y
X=df.iloc[:,0:16]
X


#  Data partation
#  Train and Test 

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
X_train.shape
X_test.shape
Y_train.shape
Y_train.shape



# Selection of best fit models
#  Model fitting for Linear regression

from sklearn.linear_model import LogisticRegression
LG=LogisticRegression()
LG.fit(X_train,Y_train)


# Model predictions

Y_train_pred=LG.predict(X_train)
Y_test_pred=LG.predict(X_test)


# Metrics

from sklearn.metrics import accuracy_score
ac1_train=accuracy_score(Y_train,Y_train_pred)
ac2_test=accuracy_score(Y_test,Y_test_pred)
print('Training accuracy score',ac1_train.round(2))
print('Test accuracy score',ac2_test.round(2))


# Cross validation for all choosen models
# K-Fold validation

from sklearn.model_selection import KFold
Kf=KFold(5)
Training_mse=[]
Test_mse=[]
for train_index,test_index in Kf.split(X):
    X_train,X_test=X.loc[train_index],X.iloc[test_index]
    Y_train,Y_test=Y.iloc[train_index],Y.iloc[test_index]
    LG.fit(X_train,Y_train)
    Y_train_pred=LG.predict(X_train)
    Y_test_pred=LG.predict(X_test)

Training_mse.append(accuracy_score(Y_train,Y_train_pred))
Test_mse.append(accuracy_score(Y_test,Y_test_pred))

print('training mean squared error:',np.mean(Training_mse).round(3))    
print('test mean squared error:',np.mean(Test_mse).round(3)) 
     

# Shrinkage methods
# Lasso Regression

from sklearn.linear_model import Lasso
LS=Lasso(alpha=1)
LS.fit(X,Y)
d1=pd.DataFrame(list(X))
d2=pd.DataFrame(LS.coef_)
df1=pd.concat([d1,d2],axis=1)
df1.columns=['names','alpha1']
df1

''' By using Lasso Regression all the coefficients becomes 0,
    so it is not possible to drop all variables,Hence verifiy it with Ridge Regression'''


    
# Ridge regression    

from sklearn.linear_model import Ridge
RR=Ridge(alpha=1)
RR.fit(X,Y)
d1=pd.DataFrame(list(X))
d2=pd.DataFrame(RR.coef_)
df1=pd.concat([d1,d2],axis=1)
df1.columns=['names','alpha1']
df1
'''By using Ridge Regression all the coefficents near to 0,
    so it is not possible to drop all variables,Hence Finilizing
    this model With all the variables'''
    


# Final Model
# Final model fitting
from sklearn.linear_model import LogisticRegression
LG=LogisticRegression()
LG.fit(X,Y)
LG.intercept_
LG.coef_


# Final model predictions

Y_pred=LG.predict(X)
Y_pred


# Metrics for the final model

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(Y,Y_pred)
cm
ac=accuracy_score(Y,Y_pred)
print('accuracy score:',(ac*100).round(2))
from sklearn.metrics import recall_score,precision_score,f1_score
print('recall_score:',(recall_score(Y,Y_pred)*100).round(2))
print('precision_score:',(precision_score(Y,Y_pred)*100).round(2))
print('f1_score:',(f1_score(Y,Y_pred)*100).round(2))


#Specificity

TN=cm[0,0]
FP=cm[1,0]
TNR=TN/(TN+FP)
print('specificity_score:',(TNR*100).round(2))


#     Probabilities 

LG.predict_proba(X).shape
LG.predict_proba(X)[:,0]   # 1-prob
LG.predict_proba(X)[:,1]    # exact probabilities


#    ROC curve

from sklearn.metrics import roc_curve,roc_auc_score
FPR,TPR,null=roc_curve(Y,LG.predict_proba(X)[:,1])



# ROC curve visualization

import matplotlib.pyplot as plt
plt.scatter(FPR,TPR)
plt.plot(FPR,TPR,color='red')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.show()


# Area under the curve

auc=roc_auc_score(Y,LG.predict_proba(X)[:,1])
print('area under curve score',(auc*100).round(2))


































