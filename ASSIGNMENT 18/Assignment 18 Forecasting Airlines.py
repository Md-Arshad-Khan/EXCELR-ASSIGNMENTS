# -*- coding: utf-8 -*-
"""

@author: arsha
"""


# Importing the data file

import pandas as pd
df=pd.read_excel("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 18 Forecasting data files/Airlines+Data.xlsx")
df


# Converting Month in dateformat

df["month"] = df.Month.dt.strftime("%b")   # Adding month column
df["year"] = df.Month.dt.strftime("%Y")          # Adding year column
df.head()


# Onehot Encoding

df_dummies=pd.DataFrame(pd.get_dummies(df['month']))
df=pd.concat([df,df_dummies],axis= 1)
df.head()
df.columns


import numpy as np

t=np.arange(1,97)
df['t'] = t       
df['t_sq'] = df['t']*df['t']                   # t square
df['log_Passengers']=np.log(df['Passengers'])     # log of Passsengers
df


# Heatmap

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=df,values="Passengers",index="year",columns="month",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g") #fmt is format of the grid values


# Boxplot for month with passengers 

plt.figure(figsize=(8,6))
sns.boxplot(x="month",y="Passengers",data=df)



# Boxplot for year with passengers 

plt.figure(figsize=(8,6))
sns.boxplot(x="year",y="Passengers",data=df)


# line plot through years

plt.figure(figsize=(12,3))
sns.lineplot(x="year",y="Passengers",data=df)


# Splitting the data into train and test

df.shape
Train = df.head(76)
Test = df.tail(20)


# Model Building

import statsmodels.formula.api as smf 

# linear model

linear_model = smf.ols('Passengers ~ t',data=Train).fit()
pred_linear = pd.Series(linear_model.predict(Test['t']))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear


#Exponential

Exp = smf.ols('log_Passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


#Quadratic 

Quad = smf.ols('Passengers~t+t_sq',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_sq"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad


#Additive seasonality 

add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea


#Additive Seasonality Quadratic 

add_sea_Quad = smf.ols('Passengers~t+t_sq+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_sq']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


#Multiplicative Seasonality

Mul_sea = smf.ols('log_Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


#Multiplicative Additive Seasonality 

Mul_Add_sea = smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 


#Compare the results 

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
type(data)

table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


''' We choose the Multiple Additive Seasonality model which has lowest RMSE which is
    11.784250.25 when compared to other models and we have created 12 dummy variables
    for this model.Hence, this is the best model for Forecasting'''











































