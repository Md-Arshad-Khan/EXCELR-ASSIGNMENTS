# -*- coding: utf-8 -*-
"""

@author: arsha
"""


# Importing the data file

import pandas as pd
df=pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 7 Clustering data files/crime_data.csv")
df


#  Shape of the data

df.shape 
df.info() 
df.describe() 


# Exploratory Data Analysis(EDA)

import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)


# Renaming the columns

df.rename(columns={'Unnamed: 0':'city'},inplace=True)
df


#       Data Transformation
#  Labelencoding for nominal data
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df['city']=LE.fit_transform(df['city'])
df


# Standardscaling for continious data

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
df['Murder']=ss.fit_transform(df[['Murder']])
df['Assault']=ss.fit_transform(df[['Assault']])
df['UrbanPop']=ss.fit_transform(df[['UrbanPop']])
df['Rape']=ss.fit_transform(df[['Rape']])
df


# Sepration of X variables 

X=df.iloc[:,1:]
X


#  DENDOGRAM
# Method = Single

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='single'))


# Agglomerative Clustering

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='single')
Y = cluster.fit_predict(X)
Y

Y_new = pd.DataFrame(Y)  # Creating Data Frame for Y variable
Y_new.value_counts() 


#    DENDOGRAM
#  Method = complete

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='complete'))


# Agglomerative Clustering

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
Y = cluster.fit_predict(X)
Y

Y_new = pd.DataFrame(Y)  #  Creating Data Frame for Y variable
Y_new.value_counts()  


#  DENDOGRAM
# Method = averagee

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(X, method='average'))


#   Agglomerative Clustering

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='average')
Y = cluster.fit_predict(X)
Y

Y_new = pd.DataFrame(Y)   # Creating Data Frame for Y Variable
Y_new.value_counts()


''' For AgglomerativeClustering we tried with all the methods that are single,complete and average. Among these methods complete linkage 
    gives the best clusers '''


# Initializing KMeans clustering

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4,n_init=20)

kmeans = kmeans.fit(X)  # Fit with inputs


# Predicting the clusters

Y = kmeans.predict(X)
Y_new = pd.DataFrame(Y)  # Creating the Data Frame for Y variable
Y_new.value_counts()  



# Sum with in centroid sum of squares 
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

''' For Kmeans clustering the 4th and 5th clusters are giving the best elbow curve .so 
    we have taken no of clusters as 4'''



# DBSCAN

X = df.iloc[:,1:].values 
X

#  Fit the DBSCAN

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=1.25, min_samples=3)
dbscan.fit(X) 

Y = dbscan.labels_
pd.DataFrame(Y).value_counts()



# Creating cluster id with dataframe

df["Cluster id"] = pd.DataFrame(Y)
df.head()


# Checking the noise points

noise_points = df[df["Cluster id"] == -1]
noise_points

# Final Data
Finaldata = df[(df["Cluster id"] == 0)| (df["Cluster id"] == 1)].reset_index(drop=True)
Finaldata

''' For DBSCAN we have taken eps=1.25 because below 1.25 we are getting more outliers
 after that we kept the noise points out and prepared a new final
    data which has other cluster ids 0's and 1's and kept them in a correct
    indexing '''













