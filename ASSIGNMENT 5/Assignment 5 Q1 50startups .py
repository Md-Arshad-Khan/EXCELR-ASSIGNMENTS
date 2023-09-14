# -*- coding: utf-8 -*-
"""

@author: arsha
"""




# Importing the data file
import numpy as np
import pandas as pd 
df =pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 5 Multi linear data files/50_Startups.csv")
df


# Renaming the variabels
df1=df.rename({'R&D Spend':'RDS','Administration':'ADMSTN','Marketing Spend':'MKS'},axis=1)
df1

# correlation analysis
df1.corr()


#                 EDA
#   scatterplot between RDS and profit

import matplotlib.pyplot as plt
plt.scatter(x=df1[['RDS']],y=df1['Profit'],color='red')
plt.xlabel('RDS')
plt.ylabel('Profit')
plt.show()
df1['RDS'].hist()  # RDS histogram
df1.boxplot(column='RDS',vert=False)  # RDS Box plot

## scatterplot between ADMSTN and profit
import matplotlib.pyplot as plt
plt.scatter(x=df1[['ADMSTN']],y=df1['Profit'],color='red')
plt.xlabel('ADMSTN')
plt.ylabel('Profit')
plt.show()
df1['ADMSTN'].hist()       # ADMSTN histogram
df1.boxplot(column='ADMSTN',vert=False)  # ADMSTN Box plot

## scatterplot between MKS and profit
import matplotlib.pyplot as plt
plt.scatter(x=df1[['MKS']],y=df1['Profit'],color='red')
plt.xlabel('MKS')
plt.ylabel('Profit')
plt.show()
df1['MKS'].hist()  # MKS histogram
df1.boxplot(column='MKS',vert=False)  # MKS Box plot


#   Finding R2 value for different models 
#   Model-1

Y=df1[['Profit']]
X=df1[['RDS']]      # MODEL 1
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y
Y_pred=LR.predict(X)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(Y,Y_pred)
R2_1=r2_score(Y,Y_pred)
print('Mean squared error:',mse.round(3))
print('Root mean sqared error:',np.sqrt(mse).round(3))
print('R square:',R2_1.round(3))


#   Model-2

Y=df1[['Profit']]
X=df1[['RDS','ADMSTN']]   # MODEL 2
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y
Y_pred=LR.predict(X)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(Y,Y_pred)
R2_2=r2_score(Y,Y_pred)
print('Mean squared error:',mse.round(3))
print('Root mean sqared error:',np.sqrt(mse).round(3))
print('R square:',R2_2.round(3))


#   Model-3

Y=df1[['Profit']]
X=df1[['MKS']]       # MODEL 3
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y
Y_pred=LR.predict(X)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(Y,Y_pred)
R2_3=r2_score(Y,Y_pred)
print('Mean squared error:',mse.round(3))
print('Root mean sqared error:',np.sqrt(mse).round(3))
print('R square:',R2_3.round(3))


#   Model-4

Y=df1[['Profit']]
X=df1[['MKS','ADMSTN']]  # MODEl 4
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y
Y_pred=LR.predict(X)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(Y,Y_pred)
R2_4=r2_score(Y,Y_pred)
print('Mean squared error:',mse.round(3))
print('Root mean sqared error:',np.sqrt(mse).round(3))
print('R square:',R2_4.round(2))


# Model-5

Y=df1[['Profit']]
X=df1[['RDS','MKS','ADMSTN']]   # MODEL 5
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y
Y_pred=LR.predict(X)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(Y,Y_pred)
R2_5=r2_score(Y,Y_pred)
print('Mean squared error:',mse.round(3))
print('Root mean sqared error:',np.sqrt(mse).round(3))
print('R square:',R2_5.round(3))

# Putting all Models RMSE and R2 in Dataframe format
d={'Models':['Model-1','Model-2','Model-3','Model-4','Model-5'],'R-Squared':[R2_1,R2_2,R2_3,R2_4,R2_5]}
R2_df=pd.DataFrame(d)
R2_df


# By taking R2 Value wee can consider Model-5 as best 
# Multicollinearity for model-5 fit

import statsmodels.formula.api as smf
model=smf.ols('Profit~ADMSTN+MKS',data=df1).fit()
model.summary()

import statsmodels.formula.api as smf
model = smf.ols('ADMSTN~MKS',data=df1).fit()
R2 = model.rsquared
VIF = 1/(1-R2)
print('Variance influence factor:',VIF)

# Residual Analysis
model.resid
model.resid.hist()

# Test for Normality of Residuals (Q-Q Plot)
import matplotlib.pyplot as plt
import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line='q')
plt.title('Q-Q plot of residuals')
plt.show()

model.fittedvalues    # predicted values
model.resid          # error values

# cheking pattern.no pattern no issues
import matplotlib.pyplot as plt
plt.scatter(model.fittedvalues,model.resid)
plt.title('Residual Plot')
plt.xlabel('Fitted values')
plt.ylabel('residual values')
plt.show()
#       Model Deletion Diagnostics
#   Detecting Influencers/Outliers

#  Cooks Distance
model_influence = model.get_influence()
(cooks, pvalue) = model_influence.cooks_distance

cooks = pd.DataFrame(cooks)

#  Plot the influencers values using stem plot
import matplotlib.pyplot as plt
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(df1)), np.round(cooks[0],5))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()

# index and value of influencer where c is more than .5
cooks[0][cooks[0]>0.5]
df.tail()

#  High Influence points
from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()

#  Leverage Cutoff
k = df1.shape[1]
n = df1.shape[0]
leverage_cutoff = (3*(k + 1)/n)
leverage_cutoff
cooks[0][cooks[0]>leverage_cutoff]
# No values are under leverage cuttoff 0.36


#            Data Transformations
#   label encoding on State variable which is Catagorical datatype

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df1["State"] = LE.fit_transform(df1["State"])
df1["State"]


# StandardScalar on ADMSTN,MKS which is a continious datatype

df1_cont = df1[['ADMSTN','MKS']]
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_cont    = SS.fit_transform(df1_cont)
SS_cont  = pd.DataFrame(SS_cont)
SS_cont.columns= ['ADMSTN','MKS']
SS_cont
df_new=pd.concat([SS_cont,df1['State'],df1['Profit']],axis=1)
df_new




# Performing Data Partition
# Splitting the Variables
Y1=df_new['Profit']
X1=df_new.iloc[:,0:2]
X1

# test and train data
from sklearn.model_selection import train_test_split
X1_train,X1_test,Y1_train,Y1_test = train_test_split(X1,Y1,train_size=0.75,random_state=42)
X1_train.shape
X1_test.shape
Y1_train
Y1_test


# Selection of few models
# Model fitting for Linear regression
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X1_train,Y1_train)


#  Model predictions
Y1_pred_train=LR.predict(X1_train)
Y1_pred_test=LR.predict(X1_test)

# Metrics
from sklearn.metrics import mean_squared_error
mse_train=np.sqrt(mean_squared_error(Y1_train,Y1_pred_train))
print('training mean squared error:',(mse_train).round(5))
mse_test=np.sqrt(mean_squared_error(Y1_test,Y1_pred_test))
print('test mean squared error:',(mse_test).round(2))


#    Cross validation for all chosen models
#  Validation set approach
Training_mse=[]
Test_mse=[]

for i in range(1,500):
    X1_train,X1_test,Y1_train,Y1_test = train_test_split(X1,Y1,train_size=0.75,random_state=i)
    LR.fit(X1_train,Y1_train)
    Y_pred_train=LR.predict(X1_train)
    Y_pred_test=LR.predict(X1_test)
    Training_mse.append(np.sqrt(mean_squared_error(Y1_train,Y1_pred_train)))
    Test_mse.append(np.sqrt(mean_squared_error(Y1_test,Y1_pred_test)))
import numpy as np
print('training mean squared error:',np.mean(Training_mse).round(3))   
print('test mean squared error:',np.mean(Test_mse).round(3)) 


#          Final Model Fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_cont    = SS.fit_transform(X1)
LR.fit(X1,Y1)
LR.intercept_
LR.coef_

#     Final Model Fitted Values
dt=pd.DataFrame({'RDS':250,'MKS':120},index=[0])
t1=LR.predict(dt)
t1












































