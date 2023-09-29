# -*- coding: utf-8 -*-
"""

@author: arsha
"""


# Salary hike

# Importing the libraries

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf

from sklearn.metrics import mean_squared_error
from math import sqrt

import warnings
warnings.filterwarnings('ignore')


# Importing the data file

salary_details = pd.read_csv('C:\PRACTISE CODING EXCELR\EXCELR ASSIGNMENTS\Assignments data files\AS 4 Simple linear assgnmt dta file/Salary_Data.csv')
salary_details


# Data Analysis

salary_details.head()
salary_details.shape
salary_details.info()
salary_details.isna().sum()


# Correlation Matrix

corr_matrix = salary_details.corr()
corr_matrix

sns.heatmap(data = corr_matrix,annot = True)
plt.show()


# Performing Assumption check
 # Outlier check using Boxplot

plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
salary_details['YearsExperience'].hist()
plt.subplot(1,2,2)
salary_details.boxplot(column = ['YearsExperience'])

plt.show()
     

plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
salary_details['Salary'].hist()
plt.subplot(1,2,2)
salary_details.boxplot(column = ['Salary'])

plt.show()

sns.distplot(salary_details['YearsExperience'])
plt.show()  # From the above histograms and boxplots we found that ther is no outliers present in the years of experience and salary data.


# Normality test using Displot

sns.distplot(salary_details['Salary'])
plt.show()          # The Normality test failed
                 


# Model building | Model training 
# Model 1  :- Without applying any transformation  

model_1 = smf.ols(formula = 'YearsExperience~Salary', data = salary_details).fit()
model_1


#coefficient
model_1.params


model_1.summary()
                 # From the Above OLS Regression Result the R-Squared value is 0.957 > 0.75 and
                     #  we can say that this Model is good to Predict Salary_hike and p-value < 0.05 and it is significant model


# Model 2 :- Applying Log transformation on Y 

model_2 = smf.ols(formula = 'Salary~np.log(YearsExperience)',data = salary_details).fit()
model_2


model_2.params


model_2.summary()


# Model 3 :- Applying Log transformation on X

model_3 = smf.ols(formula = 'np.log(Salary)~YearsExperience',data = salary_details).fit()
model_3


model_3.params


model_3.summary()


# Model 4 :- Applying Log transformation of X and Y

model_4 = smf.ols(formula = 'np.log(Salary)~np.log(YearsExperience)',data = salary_details).fit()
model_4


model_4.params

model_4.summary()


# Model 5 :- Applying Square root transformation

model_5 = smf.ols(formula = 'Salary~np.sqrt(YearsExperience)',data = salary_details).fit()
model_5

model_5.params

model_5.summary()



            # CONCLUSION = Comparing between all Models we got to know that without applying any transformation for the Model-1,
            #   we got the Higher R-squared Value i.e. 0.957 as comapare to all Model.
# Hence the Model-1 is better model to predict Salary_hike 







##########################################################################

# Delivery time 


# Importing the libraries

import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings('ignore')
     

# Importing the data file

pickup_time = pd.read_csv("C:\PRACTISE CODING EXCELR\EXCELR ASSIGNMENTS\Assignments data files\AS 4 Simple linear assgnmt dta file/delivery_time.csv")
pickup_time


# Data Analysis

pickup_time.head()

pickup_time.shape


pickup_time.info()

pickup_time.isna().sum()

pickup_time.dtypes


pickup_time1 = pickup_time.rename({'Delivery Time':'DT','Sorting Time':'ST'},axis = 1)
pickup_time1


# Correlation Matrix

corr_matrix = pickup_time1.corr()
corr_matrix
     

sns.heatmap(data = corr_matrix,annot = True)
plt.show()    # |r| > 0.8, hence there is a strong positive relation between delivery time and sorting time.

             
# Performing Assumption check 
          # Outlier check using Box plot


plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
pickup_time1['DT'].hist()
plt.subplot(1,2,2)
pickup_time1.boxplot(column = ['DT'])
plt.show()


plt.figure(figsize = (12,5))
plt.subplot(1,2,1)
pickup_time1['ST'].hist()
plt.subplot(1,2,2)
pickup_time1.boxplot(column = ['ST'])
plt.show()
            # From the above histograms and box plots, we found that there is no outliers present inside the Delivery time and sorting time variable.
            
           

# Normality test using Displot

sns.distplot(pickup_time1['DT'])
plt.show()  # Delivery time gives a positively skewed and asymmetrical distribution graph.
             # Therefore the Normality test failed, Hence the data is not normally distributed



sns.distplot(pickup_time1['ST'])
plt.show()      # Sorting time has zero skewness and Symmetrical distribution.



# Model Building | Model training 
# Model 1 :- Without applying any transformation.

time_model_1 = smf.ols(formula = 'DT~ST',data = pickup_time1).fit()
time_model_1

sns.regplot(x ='DT',y ='ST',data=pickup_time1)
plt.show()

#coefficient
time_model_1.params

time_model_1.summary()   # OLS regression results
                              # Form the above OLS regression result we got the R^2 vlaue as 0.682 that is less than 0.75. 
                                       # Therefore this model is not good to predict delivery time



# Model Testing

pred_1 = time_model_1.predict(pickup_time1.ST)
pred_1


actual_1 = pickup_time1.DT

rmse = sqrt(mean_squared_error(actual_1,pred_1))
print(rmse)


# Model 2 :- Applying the exponential transformation

time_model_2 = smf.ols(formula = 'DT~np.exp(ST)',data = pickup_time1).fit()
time_model_2
     
time_model_2.params

time_model_2.summary()


# Model 3 :- Applying the Reciprocal Transformation

time_model_3 = smf.ols(formula = 'DT~np.reciprocal(ST)',data = pickup_time1).fit()
time_model_3

time_model_3.params


time_model_3.summary()


# Model 4 :- Applying the Square transformation

time_model_4 = smf.ols(formula = 'DT~np.square(ST)',data = pickup_time1).fit()
time_model_4


time_model_4.params


time_model_4.summary()


# Model 5 :- Applying the Square root transformation

time_model_5 = smf.ols(formula = 'DT~np.sqrt(ST)',data = pickup_time1).fit()
time_model_5


time_model_5.params


time_model_5.summary()


# Model 6 :- Applying the Log transformation of X and Y.

time_model_6 = smf.ols(formula = 'np.log(DT)~np.log(ST)',data = pickup_time1).fit()
time_model_6


time_model_6.params

time_model_6.summary()


# CONCLUSION = Comparing between all Models , model_6 has Higher R-squared Value i.e. 0.772 as comapare to other Models.
# Hence the Model_6 is better Model to Predict Delivery_Time





