# -*- coding: utf-8 -*-
"""

@author: arsha
"""


#  Q1 cutlets 


# Importing data file 

import pandas as pd 
df= pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 3 Hypothesis testing data file/Cutlets.csv")
df

# Renaming the columns

df.rename(columns={'Unit A':'Unit_1','Unit B':'Unit_2'},inplace=True)

# Mean

df['Unit_1'].mean()
df['Unit_2'].mean()

# Skewness

df['Unit_1'].skew()
df['Unit_2'].skew()

# Histogram

df['Unit_1'].hist()
df['Unit_2'].hist()


# Two sample Z-Test
#--->Null Hypothesis:There is no significance difference between diameter of the Cutlets
#--->Alternative Hypothesis:There is a significance difference between diameter of the Cutlets

from scipy import stats
zcalc ,pval = stats.ttest_ind( df["Unit_1"] , df["Unit_2"]) 

print("Zcalcualted value is ",zcalc.round(4))
print("P-value is ",pval.round(4))

if pval<0.05:
    print("reject null hypothesis, Accept Alternative hypothesis")
else:
    print("accept null hypothesis, Reject Alternative hypothesis")
    
'''Here we are accepting null hypothesis and rejecting alternative hypothesis
   that means there is no significance difference between the diameter of the cutlets'''    






# Q2 Lab TAT 

# Importing the data file 

import pandas as pd 
df= pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 3 Hypothesis testing data file/LabTAT.csv")
df


# Mean

df['Laboratory 1'].mean()
df['Laboratory 2'].mean()
df['Laboratory 3'].mean()
df['Laboratory 4'].mean()

# Histogram

df['Laboratory 1'].hist()
df['Laboratory 2'].hist()
df['Laboratory 3'].hist()
df['Laboratory 4'].hist()

# Skewness

df['Laboratory 1'].skew()
df['Laboratory 2'].skew()
df['Laboratory 3'].skew()
df['Laboratory 4'].skew()


#   Anova Test

#--->H0:there is significance difference in average TAT among the different laboratories at 5% significance level.
#--->H1:there is no significance difference in average TAT among the different laboratories at 5% significance level.

import scipy.stats as stats
Fcalc, pvalue = stats.f_oneway(df["Laboratory 1"],df["Laboratory 2"],df["Laboratory 3"],df["Laboratory 4"])

Fcalc
pvalue

alpha=0.05
if (pvalue < alpha):
    print("Ho is rejected and H1 is accepted")
else:
    print("Ho is accepted and H1 is rejected")
    
''' Here Ho is rejected and H1 is accepted that means there is no significance
 difference in average TAT among the different laboratories at 5% significance level.'''








# Q3 Buyer ratio

# Importing the data file

import pandas as pd
df= pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 3 Hypothesis testing data file/BuyerRatio.csv")
df

df.info()
df.describe()


# Set the 'Gender' column as the index (optional, for better representation)

df.set_index('Observed Values', inplace=True)


# Performing the Chi-Square test

from scipy.stats import chi2_contingency
chi2_stat, p_val, dof, expected = chi2_contingency(df)


# Print the results

print("Chi-Square Statistic:", chi2_stat)
print("P-value:", p_val)
print("Degrees of Freedom:", dof)

#Ho--> Independency
#H1-->Dependency

alpha=0.05
if (p_val < alpha):
    print("Ho is rejected and H1 is accepted")
else:
    print("Ho is accepted and H1 is rejected")
    
''' Here Ho is accepted and H1 is rejected that means there is no significance
    dependency between Male and female buyers and similar in group.Hence they
    are Independent samples'''









#  Q4 Customer Order Form

# Importing the data file

import pandas as pd 
df=pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 3 Hypothesis testing data file/Costomer+OrderForm.csv")
df


# Value counts
df['Phillippines'].value_counts()
df['Indonesia'].value_counts()
df['Malta'].value_counts()
df['India'].value_counts()

# checking the datatypes
df.info()
df.describe()

# Create a contingency table using pd.crosstab

from scipy.stats import chi2_contingency
contingency_table = pd.crosstab(df['Phillippines'], [df['Indonesia'], df['Malta'], df['India']])

# Perform the chi-square test on the contingency table



chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)

print("Chi-square statistic:", chi2_stat)
print("P-value:", p_val)
print("Degrees of freedom:", dof)
print("Expected frequencies:\n", expected)

#Ho--> Independency
#H1-->Dependency

alpha=0.05
if (p_val < alpha):
    print("Ho is rejected and H1 is accepted")
else:
    print("Ho is accepted and H1 is rejected")
    
''' Here Ho is accepted and H1 is rejected that means there is no significance
    dependency between countries and similar groups.Hence they
    are Independent'''

















