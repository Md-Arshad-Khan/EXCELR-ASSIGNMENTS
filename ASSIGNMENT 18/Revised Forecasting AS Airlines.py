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


# Importing the Data file

passengers_data = pd.read_excel('C:\PRACTISE CODING EXCELR\EXCELR ASSIGNMENTS\Assignments data files\AS 18 Forecasting data files/Airlines+Data.xlsx')
passengers_data
     

# Data Analysis

passengers_data.head()

passengers_data.shape

passengers_data.info()

passengers_data.isna().sum()

passengers_data.describe()

passengers_data.dtypes

passengers_data.columns

passengers_data.set_index('Month', inplace = True)
passengers_data.head()
 

# Visualizing using lineplot for passengers

plt.figure(figsize = (8,5))

plt.xlabel("Date")
plt.ylabel("Number of air passengers")
ax = plt.axes()
ax.set_facecolor("black")

plt.plot(passengers_data['Passengers'], linewidth = 2)

plt.show()


# Visualizing using histogram

ax = plt.axes()
ax.set_facecolor("black")

passengers_data['Passengers'].hist(figsize = (8,5))

plt.show()


# Visualizing using lagplot

from pandas.plotting import lag_plot

ax = plt.axes()
ax.set_facecolor("black")

lag_plot(passengers_data['Passengers'])

plt.show()


# Visualizing using TSA plot

import statsmodels.graphics.tsaplots as tsa_plots

tsa_plots.plot_acf(passengers_data['Passengers'],lags = 12)

tsa_plots.plot_pacf(passengers_data['Passengers'],lags = 12)

plt.show()


# Data Driven Forecasting Models

from statsmodels.tsa.holtwinters import SimpleExpSmoothing 
from statsmodels.tsa.holtwinters import Holt 
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# Splitting Data 

Train = passengers_data.head(84)
Test = passengers_data.tail(12)


# Moving Average Method

plt.figure(figsize = (15,5))
passengers_data['Passengers'].plot(label = "org")

for i in range(2,8,2):
    passengers_data['Passengers'].rolling(i).mean().plot(label = str(i))
    
plt.legend(loc = 'best')
plt.show()


# Time series decomposition plot

from statsmodels.tsa.seasonal import seasonal_decompose

ts_decompose = seasonal_decompose(passengers_data.Passengers,period = 12)
ts_decompose.plot()
plt.show()


# Evaluation Metric RMSE

def RMSE(org, pred):
    rmse = np.sqrt(np.mean((np.array(org)-np.array(pred))**2))
    return rmse
     
import warnings
warnings.filterwarnings('ignore')


# Simple Exponential Method

simple_model = SimpleExpSmoothing(Train["Passengers"]).fit()
pred_simple_model = simple_model.predict(start = Test.index[0],end = Test.index[-1])

rmse_simple_model = RMSE(Test.Passengers, pred_simple_model)
print('RMSE Value of Simple Exponential :',rmse_simple_model)


# Holt Method

holt_model = Holt(Train["Passengers"]).fit()
pred_holt_model = holt_model.predict(start = Test.index[0],end = Test.index[-1])

rmse_holt_model = RMSE(Test.Passengers, pred_holt_model)
print('RMSE Value of Holt :',rmse_holt_model)

rmse_holt_model = 58.57776


# Holts winter exponential smoothing with additive seasonality and additive trend :

holt_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal = "add",trend = "add",seasonal_periods = 4).fit()
pred_holt_add_add = holt_model_add_add.predict(start = Test.index[0],end = Test.index[-1])

rmse_holt_add_add_model = RMSE(Test.Passengers, pred_holt_add_add)
print('RMSE Value of Holts add and add :',rmse_holt_add_add_model)

rmse_holt_add_add_model =62.91998
     

# Holts winter exponential smoothing with multiplicative seasonality and additive trend

holt_model_multi_add = ExponentialSmoothing(Train["Passengers"],seasonal = "mul",trend = "add",seasonal_periods = 4).fit() 
pred_holt_multi_add = holt_model_multi_add.predict(start = Test.index[0],end = Test.index[-1])

rmse_holt_model_multi_add_model = RMSE(Test.Passengers, pred_holt_multi_add)
print('RMSE Value of Holts Multi and add :',rmse_holt_model_multi_add_model)

rmse_holt_model_multi_add_model = 64.6126


# Model based Forecasting Methods
#   Data preprocessing for models

passengers_data_1 = passengers_data.copy()
passengers_data_1.head()

passengers_data_1["t"] = np.arange(1,97)
passengers_data_1["t_squared"] = passengers_data_1["t"]*passengers_data_1["t"]

passengers_data_1["log_psngr"] = np.log(passengers_data_1["Passengers"])
passengers_data_1.head()


# Splitting Data

Train = passengers_data_1.head(84)
Test = passengers_data_1.tail(12)


# Linear Model

import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers~t',data = Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))

rmse_linear_model = RMSE(Test['Passengers'], pred_linear)
print('RMSE Value of Linear :',rmse_linear_model)

rmse_linear_model = 53.1992


# Exponential Model

Exp_model = smf.ols('log_psngr~t',data = Train).fit()
pred_Exp = pd.Series(Exp_model.predict(pd.DataFrame(Test['t'])))

rmse_Exp_model = RMSE(Test['Passengers'], np.exp(pred_Exp))
print('RMSE Value of Exponential :',rmse_Exp_model)

rmse_Exp_model =46.0573


# Quadratic Model

Quad_model= smf.ols('Passengers~t+t_squared',data = Train).fit()
pred_Quad = pd.Series(Quad_model.predict(Test[["t","t_squared"]]))

rmse_Quad_model = RMSE(Test['Passengers'], pred_Quad)
print('RMSE Value of Quadratic :',rmse_Quad_model)

rmse_Quad_model = 48.05188


# ARIMA Model

series = passengers_data.copy()
series


# Seprating a validation datset

split_point = len(series) - 12
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', header = False)
validation.to_csv('validation.csv', header = False)


# Evaluate a base Model

from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt


train = read_csv('dataset.csv', header = None, index_col = 0, parse_dates = True, squeeze = True)

X = train.values
X = X.astype('float32')
train_size = int(len(X) * 0.715)
train, test = X[0:train_size], X[train_size:]

print(train.shape)
print(test.shape)
     

# Walk Forward validation

history = [x for x in train]
predictions = list()

for i in range(len(test)):
    yhat = history[-1]
    predictions.append(yhat)
# observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
     
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE Value : %.3f' % rmse)

rmse_Persistence_model = 29.058


# CONCLUSION

list = [['Simple Exponential Method',rmse_simple_model], ['Holt method',rmse_holt_model],
          ['Holt exp smoothing add',rmse_holt_add_add_model],['Holt exp smoothing multi',rmse_holt_model_multi_add_model],
          ['Linear Model',rmse_linear_model],['Exponential model',rmse_Exp_model],['Quadratic model',rmse_Quad_model],
          ['Persistence/ Base model', rmse_Persistence_model]]


df = pd.DataFrame(list, columns = ['Model', 'RMSE_Value']) 
df












