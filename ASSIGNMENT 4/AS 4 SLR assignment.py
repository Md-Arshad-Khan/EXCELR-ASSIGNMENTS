# -*- coding: utf-8 -*-
"""

@author: arsha
"""


#  Q1

############################################################
# #  EXTRACTING DATA FILE 
import numpy as np
import pandas as pd 
df= pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 4 Simple linear assgnmt dta file/delivery_time.csv")
df


X= df["Sorting Time"]
X

Y= df["Delivery Time"]
Y

df1=df["Sorting Time"]
df1.ndim
df2=df["Delivery Time"]
df2.ndim


X=np.array(df1).reshape(-1,1)
Y=np.array(df2).reshape(-1,1)

# #  SCATTER PLOT 
import matplotlib.pyplot as plt 
plt.scatter(X,Y)
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()


# Model Fitting 


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_  # (Bo) BITa not
LR.coef_        # m (slope)


# ======================= MODEL PREDICTED VALUES ======================
Y_pred =LR.predict(X)
Y_pred


df.corr()

## model plotted with orginal values
import matplotlib.pyplot as plt
plt.scatter(x=df1,y=df2,color='red')
plt.scatter(x=df1,y=Y_pred,color='blue')
plt.plot(df1,Y_pred,color='black')
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

# calculating the error
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print('Mean squared error:',mse.round(4))
print('Root mean sqared error:',np.sqrt(mse).round(4))

#  model prediction
P1= np.array([[7]])
LR.predict(P1)

















# Q2

############################################################
# #  EXTRACTING DATA FILE 
import numpy as np
import pandas as pd 
df= pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 4 Simple linear assgnmt dta file/Salary_Data.csv")
df


X= df["YearsExperience"]
X

Y= df["Salary"]
Y

df1=df["YearsExperience"]
df1.ndim
df2=df["Salary"]
df2.ndim


X=np.array(df1).reshape(-1,1)
Y=np.array(df2).reshape(-1,1)

# #  SCATTER PLOT 
import matplotlib.pyplot as plt 
plt.scatter(X,Y)
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()


# Model Fitting 


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)
LR.intercept_  # (Bo) BITa not
LR.coef_        # m (slope)


# ======================= MODEL PREDICTED VALUES ======================
Y_pred =LR.predict(X)
Y_pred


df.corr()

## model plotted with orginal values
import matplotlib.pyplot as plt
plt.scatter(x=df1,y=df2,color='red')
plt.scatter(x=df1,y=Y_pred,color='blue')
plt.plot(df1,Y_pred,color='black')
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

# calculating the error
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y,Y_pred)
print('Mean squared error:',mse.round(4))
print('Root mean sqared error:',np.sqrt(mse).round(4))

# model prediction
P1= np.array([[12]])
LR.predict(P1)





























