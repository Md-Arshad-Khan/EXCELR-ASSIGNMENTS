# -*- coding: utf-8 -*-
"""

@author: arsha
"""

# Importing the libraries

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# Importing the data file

sales_data = pd.read_excel('C:\PRACTISE CODING EXCELR\EXCELR ASSIGNMENTS\Assignments data files\AS 18 Forecasting data files/CocaCola_Sales_Rawdata.xlsx')
sales_data


# Data Analysis

sales_data.head()

sales_data.shape

sales_data.info()

sales_data.isna().sum()


sales_data.describe()

sales_data.dtypes

sales_data.columns

temp = sales_data.Quarter.str.replace(r'(Q\d)_(\d+)', r'19\2-\1')
sales_data['quater'] = pd.to_datetime(temp).dt.strftime('%b-%Y')
sales_data.head()

sales_data = sales_data.drop(['Quarter'], axis = 1)
sales_data.reset_index(inplace=True)
sales_data['quater'] = pd.to_datetime(sales_data['quater'])
sales_data = sales_data.set_index('quater')
sales_data.head()


# Visualizing using line plot for sales

sales_data['Sales'].plot(figsize = (15, 6))
plt.show()


# Moving Average method

for i in range(2,10,2):
    sales_data["Sales"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)
plt.show()
     

# Time series decomposition plot

from statsmodels.tsa.seasonal import seasonal_decompose

ts_add = seasonal_decompose(sales_data.Sales,model = "additive")
fig = ts_add.plot()
plt.show()

ts_mul = seasonal_decompose(sales_data.Sales,model = "multiplicative")
fig = ts_mul.plot()
plt.show()


# Visualizing using TSA plot

import statsmodels.graphics.tsaplots as tsa_plots

tsa_plots.plot_acf(sales_data.Sales)

tsa_plots.plot_pacf(sales_data.Sales)

plt.show()
     

# Evaluation Metric RMSE

from statsmodels.tsa.holtwinters import SimpleExpSmoothing 
from statsmodels.tsa.holtwinters import Holt 
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import warnings
warnings.filterwarnings('ignore')


def RMSE(org, pred):
    rmse = np.sqrt(np.mean((np.array(org)-np.array(pred))**2))
    return rmse


# Splitting the Data 

Train = sales_data.head(30)
Test = sales_data.tail(12)


# Simple exponential method

simple_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_simple_model = simple_model.predict(start = Test.index[0],end = Test.index[-1])

rmse_simple_model = RMSE(Test.Sales, pred_simple_model)
print('RMSE Value of Simple Exponential :',rmse_simple_model)

rmse_simple_model=860.8833
     

# Holt method

holt_model = Holt(Train["Sales"]).fit()
pred_holt_model = holt_model.predict(start = Test.index[0],end = Test.index[-1])

rmse_holt_model = RMSE(Test.Sales, pred_holt_model)
print('RMSE Value of Holt :',rmse_holt_model)

rmse_holt_model=518.1409

#  Holts winter exponential smoothing with additive seasonality and additive trend 

holt_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal = "add",trend = "add",seasonal_periods = 4).fit()
pred_holt_add_add = holt_model_add_add.predict(start = Test.index[0],end = Test.index[-1])

rmse_holt_add_add_model = RMSE(Test.Sales, pred_holt_add_add)
print('RMSE Value of Holts add and add :',rmse_holt_add_add_model)

rmse_holt_add_add_model=231.4577


# Holts winter exponential smoothing with multiplicative seasonality and additive trend

holt_model_multi_add = ExponentialSmoothing(Train["Sales"],seasonal = "mul",trend = "add",seasonal_periods = 4).fit() 
pred_holt_multi_add = holt_model_multi_add.predict(start = Test.index[0],end = Test.index[-1])


rmse_holt_model_multi_add_model = RMSE(Test.Sales, pred_holt_multi_add)
print('RMSE Value of Holts Multi and add :',rmse_holt_model_multi_add_model)


rmse_holt_model_multi_add_model = 194.7006


# Model based Forecasting Methods
# Data preprocessing for models:


sales_data_1 = pd.read_excel('C:\PRACTISE CODING EXCELR\EXCELR ASSIGNMENTS\Assignments data files\AS 18 Forecasting data files/CocaCola_Sales_Rawdata.xlsx')
sales_data_1.head()


sales_data_2 = pd.get_dummies(sales_data_1, columns = ['Quarter'])
sales_data_2.columns = ['Sales','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1','Q1',
                        'Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2','Q2',
                        'Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q3','Q3',
                        'Q4','Q4','Q4','Q4','Q4','Q4','Q4','Q4','Q4','Q4']
sales_data_2.head()



t = np.arange(1,43)
sales_data_2['t'] = t
sales_data_2['t_squared'] = sales_data_2['t']*sales_data_2['t']
log_Sales = np.log(sales_data_2['Sales'])

sales_data_2['log_Sales'] = log_Sales
sales_data_2.head()


# Splitting the Data

train, test = np.split(sales_data_2, [int(.67 *len(sales_data_2))])


# Linear Model

import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data = train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(test['t'])))


rmse_linear_model = RMSE(test['Sales'], pred_linear)
print('RMSE Value of Linear :',rmse_linear_model)


rmse_linear_model=750.4020
  

# Exponential Model
   
Exp_model = smf.ols('log_Sales~t',data = train).fit()
pred_Exp = pd.Series(Exp_model.predict(pd.DataFrame(test['t'])))

rmse_Exp_model = RMSE(test['Sales'], np.exp(pred_Exp))
print('RMSE Value of Exponential :',rmse_Exp_model)

rmse_Exp_model=588.1405
     

# Quadratic Model

Quad_model= smf.ols('Sales~t+t_squared',data = train).fit()
pred_Quad = pd.Series(Quad_model.predict(test[["t","t_squared"]]))

rmse_Quad_model = RMSE(test['Sales'], pred_Quad)
print('RMSE Value of Quadratic :',rmse_Quad_model)

rmse_Quad_model = 783.7297


# Additive Model

additive_model =  smf.ols('Sales~ Q1+Q2+Q3+Q4',data = train).fit()
pred_additive = pd.Series(additive_model.predict(pd.DataFrame(test[['Q1','Q2','Q3','Q4']])))

rmse_additive_model = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_additive))**2))
print('RMSE Value of Additive :',rmse_additive_model)

rmse_additive_model = 1869.7188


# Additive Linear Model     

additive_linear_model = smf.ols('Sales~t+Q1+Q2+Q3+Q4',data = train).fit()
pred_additive_linear = pd.Series(additive_linear_model.predict(pd.DataFrame(test[['t','Q1','Q2','Q3','Q4']])))

rmse_additive_linear = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_additive_linear))**2))
print('RMSE Value of Additive Linear :',rmse_additive_linear)

rmse_additive_linear= 596.1526


# Additive Quadratic Model

additive_quad_model = smf.ols('Sales~t+t_squared+Q1+Q2+Q3+Q4',data = train).fit()
pred_additive_quad = pd.Series(additive_quad_model.predict(pd.DataFrame(test[['t','t_squared','Q1','Q2','Q3','Q4']])))

rmse_additive_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(pred_additive_quad))**2))
print('RMSE Value of Additive Quadratic :',rmse_additive_quad)

rmse_additive_quad=412.1144


# Multi Linear Model

multi_linear_model = smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data = train).fit()
pred_multi_linear = pd.Series(multi_linear_model.predict(pd.DataFrame(test[['t','Q1','Q2','Q3','Q4']])))


rmse_multi_linear = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_multi_linear)))**2))
print('RMSE Value of Multi Linear :',rmse_multi_linear)

rmse_multi_linear = 5359.6879


# Multi Quadratic Model

multi_quad_model = smf.ols('log_Sales~t+t_squared+Q1+Q2+Q3+Q4',data = train).fit()
pred_multi_quad = pd.Series(multi_quad_model.predict(test[['t','t_squared','Q1','Q2','Q3','Q4']]))

rmse_multi_quad = np.sqrt(np.mean((np.array(test['Sales'])-np.array(np.exp(pred_multi_quad)))**2))
print('RMSE Value of Multi Quadratic :',rmse_multi_quad)

rmse_multi_quad = 3630.5619


# ARIMA Model

series = sales_data_1.copy()
series


# Seprating a validation Dataset 

split_point = len(series) - 12
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', header = False)
validation.to_csv('validation.csv', header = False)


# Evaluate a base Model

X = sales_data_1['Sales'].values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]


# Walk Forward Validation 

from sklearn.metrics import mean_squared_error
from math import sqrt


history = [x for x in train]
predictions = list()

for i in range(len(test)):
    yhat = history[-1]
    predictions.append(yhat)
# observation
    obs = test[i]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE Value : %f' % rmse)



rmse_Persistence_model = 565.7799


# CONCLUSION

list = [['Simple Exponential Method',rmse_simple_model], ['Holt method',rmse_holt_model],
          ['Holt exp smoothing add',rmse_holt_add_add_model],['Holt exp smoothing multi',rmse_holt_model_multi_add_model],
          ['Linear Model',rmse_linear_model],['Exponential model',rmse_Exp_model],['Quadratic model',rmse_Quad_model],
          ['Additive Model',rmse_additive_model],['Additive Linear Model',rmse_additive_linear],
          ['Additive Qudratic Model',rmse_additive_quad],['Muli Linear Model',rmse_multi_linear],
          ['Multi Quadratic Model',rmse_multi_quad],
          ['Persistence/ Base model', rmse_Persistence_model]]

df = pd.DataFrame(list, columns = ['Model', 'RMSE_Value']) 
df

















