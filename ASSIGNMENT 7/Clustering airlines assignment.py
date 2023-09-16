# -*- coding: utf-8 -*-
"""
@author: arsha
"""

# Importing the data file

import pandas as pd 
df= pd.read_excel("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 7 Clustering data files/EastWestAirlines.xlsx",sheet_name="data")
df

df.head()
df.tail()
df.columns

# shape of the data
df.shape

df.info() 
df.describe() 
df.columns



# Exploratory Data Analysis (EDA)

import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)


#  Renamin the columns of the data  

df.rename(columns={'ID#':'ID','Award?':'Award'},inplace=True)
df.columns


#    Data Transformation
#  Standardscaling 

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
for column in df.columns:
    if df[column].dtype=='object':
        continue
    df[column]=ss.fit_transform(df[[column]])


# Sepration of X variables 

X=df.iloc[:,1:]
X


#       DENDOGRAM
#     Method=single

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='single'))


# Agglomerative Clustering

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='single')
Y = cluster.fit_predict(X)
Y

Y_new = pd.DataFrame(Y)  # Creating Data Frame for Y variable
Y_new.value_counts() 


# DENDOGRAM
# Method = complete

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(30,15))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='complete'))


# Agglomerative Clustering

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
Y = cluster.fit_predict(X)
Y

Y_new = pd.DataFrame(Y)  # Creating Data Frame for Y variable
Y_new.value_counts()  



# DENDOGRAM
# Method = averagee

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(30,15))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='average'))


# Agglomerative Clustering

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')
Y = cluster.fit_predict(X)
Y

Y_new = pd.DataFrame(Y)  # Creating Data Frame for Y variable
Y_new.value_counts()  

''' For Agglomerative Clustering We tried all the three different methods that are single, complete and average. Amoung these methods complete linkage 
    gives best clusters '''


#   Initializing KMeans clustering

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5,n_init=20)

kmeans = kmeans.fit(X)    # Fit with inputs


# Predicting the clusters

Y = kmeans.predict(X)
Y_new = pd.DataFrame(Y) # Creating Data Frame for Y variable
Y_new.value_counts()  


# Total with in centroid sum of squares 

kmeans.inertia_

clust = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(X)
    clust.append(kmeans.inertia_)


# Elbow method 

plt.scatter(x=range(1,11), y=clust,color='red')
plt.plot(range(1,11), clust,color='black')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertial values')
plt.show()



# DBSCAN

X = df.iloc[:,1:].values 
X

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=3, min_samples=5)
dbscan.fit(X)  # Fitting DBSCAN

Y = dbscan.labels_
pd.DataFrame(Y).value_counts()


# Creating cluster id with DataFrame
df["Cluster id"] = pd.DataFrame(Y)
df.head()


# Checking the noise points
noise_points = df[df["Cluster id"] == -1]
noise_points


# Final data
Finaldata = df[(df["Cluster id"] == 0)| (df["Cluster id"] == 1)
               |(df["Cluster id"] == 2)].reset_index(drop=True)
Finaldata

''' For DBSCAN we have taken eps=3 because below 3 we are getting more noise points.  
    We left the noise points out and prepared a new final data which has other
    cluster ids 0's,1's and 2's and kept them in a correct indexing '''


































