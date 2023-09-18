# -*- coding: utf-8 -*-
"""

@author: arsha
"""

# Importing the data file

import pandas as pd
df =pd.read_excel("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 18 Forecasting data files/CocaCola_Sales_Rawdata.xlsx")
df


# Converting Quarter column into 4 Quarter Column

df['quarter'] = 0
for i in range(42):
    p=df['Quarter'][i]
    df['quarter'][i]=p[0:2]
    

# Heatmap

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,8))
heatmap_quarters = pd.pivot_table(data=df,values="Sales",index="quarter",fill_value=0)
sns.heatmap(heatmap_quarters,annot=True,fmt="g") #fmt is format of the grid values
  

# Boxplot

plt.figure(figsize=(8,6))
sns.boxplot(x="quarter",y="Sales",data=df)


# Line plot through the Quarters

plt.figure(figsize=(12,3))
sns.lineplot(x="quarter",y="Sales",data=df)


# Onehot Encoding

df_dummies=pd.DataFrame(pd.get_dummies(df['quarter']),columns=['Q1','Q2','Q3','Q4'])
df=pd.concat([df,df_dummies],axis= 1)
df

import numpy as np

t=np.arange(1,43)
df['t'] = t                         
df['t_sq'] = df['t']*df['t']  
df['log_sales']=np.log(df['Sales']) 
df                                   # Added t,t2 and log columns


# Split the data into train and test

df.shape
Train = df.head(31)
Test = df.tail(11)


# Model Building

import statsmodels.formula.api as smf 

# linear model

linear_model = smf.ols('Sales ~ t',data=Train).fit()
pred_linear = pd.Series(linear_model.predict(Test['t']))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear


# Exponential

Exp = smf.ols('log_sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# Quadratic 

Quad = smf.ols('Sales~t+t_sq',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_sq"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad


# Additive seasonality 

add_sea = smf.ols('Sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3','Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea


#  Additive Seasonality Quadratic 

add_sea_Quad = smf.ols('Sales~t+t_sq+Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','Q4','t','t_sq']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# Multiplicative Seasonality

Mul_sea = smf.ols('log_sales~Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# Multiplicative Additive Seasonality 

Mul_Add_sea = smf.ols('log_sales~t+Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 


# Comparing the results 

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
type(data)

table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


''' we have chosen Multiple Additive Seasonality model which has lowest RMSE which is
    427.25 when compared to other models and created 4 dummy variables
    for this model.Hence can conclude this is the best model for Forcasting'''








