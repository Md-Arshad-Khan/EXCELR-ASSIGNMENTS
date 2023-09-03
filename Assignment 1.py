# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 19:30:30 2023

@author: arsha
"""

# ASSIGNMENT 1 Q7


import pandas as pd 
df=pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/Assignment 1 data file/Q7.csv")
df

df.describe()


# ========================

df["Points"].describe()
df["Points"].median()


Range = df["Points"].max() - df["Points"].min()
Range


df["Points"].var()
df["Points"].mode()


# ==========


df["Score"].describe()
df["Score"].median()


Range = df["Score"].max() - df["Score"].min()
Range


df["Score"].var()
df["Score"].mode()


# ===========

df["Weigh"].describe()
df["Weigh"].median()


Range = df["Weigh"].max() - df["Weigh"].min()
Range


df["Weigh"].var()
df["Weigh"].mode()




# ASSIGNMENT 1 Q9 A and B

import pandas as pd 
df= pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/Assignment 1 data file/Q9_a.csv")
df


df['speed'].hist()
df['speed'].skew()
df['speed'].kurtosis()




df['dist'].hist()
df['dist'].skew()
df['dist'].kurtosis()






# Q9 B 


import pandas as pd 
df= pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/Assignment 1 data file/Q9_b.csv")
df

df['SP'].hist()
df["SP"].skew()
df["SP"].kurtosis()



df['WT'].hist()
df["WT"].skew()
df["WT"].kurtosis()



# Q11

from scipy import stats
mean = 200
std  = 30
# confidence interval for 94%


df_ci = stats.norm.interval(0.94,
                                 loc=200,
                                 scale=30)

print ("I am 94% confident that population mean weight lies under:", df_ci)



# confidence interval for 96%


df_ci = stats.norm.interval(0.96,
                                 loc=200,
                                 scale=30)

print ("I am 96% confident that population mean weight lies under:", df_ci)



# confidence interval for 98%


df_ci = stats.norm.interval(0.98,
                                 loc=200,
                                 scale=30)

print ("I am 98% confident that population mean weight lies under:", df_ci)




# Q12
import pandas as pd

marks = [34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]
df= pd.DataFrame(marks)

df.mean()
df.median()
df.var()
df.std()




# Q 20 



import pandas as pd 
df= pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/Assignment 1 data file/cars.csv")
df

len(df)

df[df["MPG"]>38]

df[df["MPG"]>38].count()

(df[df['MPG']>38].count ()/len (df)).round(3)*100


df[df["MPG"]<40]

df[df["MPG"]<40].count()

(df[df['MPG']<40].count ()/len (df)).round(3)*100



df[(df['MPG']>20) & (df['MPG']<50)].count()

(df[(df['MPG']>20) & (df['MPG']<50)].count()/len (df)).round(3)*100




df["MPG"].mean()
df["MPG"].std()

from scipy.stats import norm

nd=norm(34.42207572802469,9.131444731795982)

Z1= nd.cdf(50)

Z2= nd.cdf(20)


Z1-Z2








# Q 21 
# a


import pandas as pd 
df= pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/Assignment 1 data file/cars.csv")
df
df["MPG"].hist()
df["MPG"].skew()

# Check whetherthe MPG of cars is following normal distribution or not 

from scipy.stats import shapiro
calc,p = shapiro(df["MPG"])
calc
p

alpha =0.05

if (p<alpha):
    print("Ho is rejected and H1 is accepted")
else:
    print ("Ho  is accepted and H1 is rejected")

# Ho is accepted : Data is normal 
# H1 is accepted : Data is not normal 


# b

import pandas as pd 
df= pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/Assignment 1 data file/wc-at.csv")
df

import matplotlib.pyplot as plt 
df["AT"].hist()
df["Waist"].hist()

df["AT"].skew()
df["Waist"].skew()

# Check whether the adipose tissue data follows normal distribution or not 

from scipy.stats import shapiro 
calc,p = shapiro(df["AT"])
calc 
p

alpha = 0.05

if (p < alpha):
    print ("Ho is rejected and H1 is accepted")
else:
    print ("Ho is accepted and H1 is rejected")
    
# Check whether the waist data follows normal distribution or not 

from scipy.stats import shapiro 
calc,p = shapiro(df["Waist"])
calc 
p

alpha = 0.05

if (p < alpha):
    print ("Ho is rejected and H1 is accpeted")
else:
    print ("Ho is accepted and H1 is rejected")
    

# Ho is accepted : Data is normal 
# H1 is accepted : Data is not normal 




# Q 22

from scipy.stats import norm

# Z-score for 90% confidence interval 

confidence_level = 0.90

z_score = norm.ppf(1 - (1 - confidence_level) / 2)

print(f"Z-score for a {int(confidence_level * 100)}% confidence interval: {z_score:.4f}")




# Z-score for 94% confidence interval 

confidence_level = 0.94

z_score = norm.ppf(1 - (1 - confidence_level) / 2)

print(f"Z-score for a {int(confidence_level * 100)}% confidence interval: {z_score:.4f}")



# Z-score for 60% confidence interval 

confidence_level = 0.60

z_score = norm.ppf(1 - (1 - confidence_level) / 2)

print(f"Z-score for a {int(confidence_level * 100)}% confidence interval: {z_score:.4f}")






#Q 23

# t score for 95% confidence interval

from scipy.stats import t

confidence_level = 0.95
sample_size = 25
degrees_of_freedom = sample_size - 1

# Calculate t-score for the given confidence level and degrees of freedom
t_score = t.ppf(1 - (1 - confidence_level) / 2, df=degrees_of_freedom)

print(f"t-score for a {int(confidence_level * 100)}% confidence interval with {sample_size} sample size: {t_score:.4f}")



# t score for 96% confidence interval

from scipy.stats import t

confidence_level = 0.96
sample_size = 25
degrees_of_freedom = sample_size - 1

# Calculate t-score for the given confidence level and degrees of freedom
t_score = t.ppf(1 - (1 - confidence_level) / 2, df=degrees_of_freedom)

print(f"t-score for a {int(confidence_level * 100)}% confidence interval with {sample_size} sample size: {t_score:.4f}")



# t score for 99% confidence interval

from scipy.stats import t

confidence_level = 0.99
sample_size = 25
degrees_of_freedom = sample_size - 1

# Calculate t-score for the given confidence level and degrees of freedom
t_score = t.ppf(1 - (1 - confidence_level) / 2, df=degrees_of_freedom)

print(f"t-score for a {int(confidence_level * 100)}% confidence interval with {sample_size} sample size: {t_score:.4f}")








# Q 24

import math
import scipy.stats as stats

x_bar = 260
mu = 270
s = 90
n = 18

t_score = (x_bar - mu) / (s / math.sqrt(n))
df = n - 1

p_value = stats.t.cdf(t_score, df)
print("The probability that 18 randomly selected bulbs would have an average life of no more than 260 days:", p_value)





















