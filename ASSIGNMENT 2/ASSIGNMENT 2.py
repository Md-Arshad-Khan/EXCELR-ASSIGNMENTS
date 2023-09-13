# -*- coding: utf-8 -*-
"""

@author: arsha
"""

import pandas as pd

# Data given
data = {'Name of company': ['Allied Signal', 'Bankers Trust', 'General Mills', 'ITT Industries',
                        'J.P.Morgan & Co.', 'Lehman Brothers', 'Marriott', 'MCI',
                        'Merrill Lynch', 'Microsoft', 'Morgan Stanley', 'Sun Microsystems',
                        'Travelers', 'US Airways', 'Warner-Lambert'],
    'Measure X': ['24.23%', '25.53%', '25.41%', '24.14%', '29.62%', '28.25%', '25.81%', '24.39%',
                  '40.26%', '32.95%', '91.36%', '25.99%', '39.42%', '26.71%', '35.00%']}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Convert 'Measure X' to numeric values (remove the % sign and convert to float)
df['Measure X'] = df['Measure X'].str.rstrip('%').astype(float)

import matplotlib.pyplot as plt

# Plot the 'Measure X' values
plt.figure(figsize=(10, 6))
plt.bar(df['Name of company'], df['Measure X'])
plt.xticks(rotation=90)
plt.xlabel('Name of Company')
plt.ylabel('Measure X')
plt.title('Measure X for Companies')
plt.tight_layout()
plt.show()

# Calculate the mean, standard deviation, and variance
mean_x = df['Measure X'].mean()
std_x = df['Measure X'].std()
var_x = df['Measure X'].var()

print(f"Mean (μ): {mean_x:.2f}")
print(f"Standard Deviation (σ): {std_x:.2f}")
print(f"Variance (σ^2): {var_x:.2f}")



# Calculate the first quartile (Q1) and third quartile (Q3)
q1 = df['Measure X'].quantile(0.25)
q3 = df['Measure X'].quantile(0.75)

# Calculate the interquartile range (IQR)
IQR = q3 - q1

# Define the lower whisker and upper whisker for outliers
lower_whisker = q1 - 1.5 * IQR
upper_whisker = q3 + 1.5 * IQR

# Find outliers
outliers = df[(df['Measure X'] < lower_whisker) | (df['Measure X'] > upper_whisker)]

print(outliers)