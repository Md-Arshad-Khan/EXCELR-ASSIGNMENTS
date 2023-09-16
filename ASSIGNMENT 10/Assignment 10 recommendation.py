# -*- coding: utf-8 -*-
"""

@author: arsha
"""

# Importing the data file

import pandas as pd
df=pd.read_csv("C:/PRACTISE CODING EXCELR/EXCELR ASSIGNMENTS/Assignments data files/AS 10 recommendation data files/book.csv",encoding="latin")
df
df.columns


# Drop the index Column

df.drop(columns='Unnamed: 0',inplace=True)
df


# Renaming the columns

df = df.rename(columns={'User.ID': 'UserId', 'Book.Title': 'Book_title', 'Book.Rating': 'Book_Rating'})
df=df.sort_values('UserId')    # Sorting  the columns
df


# Length of unique userId and booktitles

len(df.UserId.unique())
len(df.Book_title.unique())

df['Book_Rating'].value_counts() 
df['Book_Rating'].hist() ## Histogram 


# pivot Table

user_books = df.pivot_table(index='UserId',
                                 columns='Book_title',
                                 values='Book_Rating')

user_books


#Impute those NaNs with 0 values

user_books.fillna(0, inplace=True)
user_books


# Pairwise  distance

from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
user_sim = 1 - pairwise_distances(user_books.values,metric='cosine')
user_sim



#Store the results in a dataframe

user_E_df = pd.DataFrame(user_sim)

#Set the index and column names to user ids 
user_E_df.index   = df.UserId.unique()
user_E_df.columns = df.UserId.unique()
user_E_df

import numpy as np

np.fill_diagonal(user_sim, 0) ## Filling the diagonal with 0
user_E_df

user_E_df.idxmax(axis=1)[0:50]


# based on the common interests we will recommend the producta to other users
df[(df['UserId']==86) | (df['UserId']==276780)]
df[(df['UserId']==160) | (df['UserId']==899)]
df[(df['UserId']==81) | (df['UserId']==8)]
df[(df['UserId']==85) | (df['UserId']==1211)]
df[(df['UserId']==83) | (df['UserId']==276861)]
df[(df['UserId']==82) | (df['UserId']==882)]
df[(df['UserId']==53) | (df['UserId']==1996)]
df[(df['UserId']==51) | (df['UserId']==3757)]
df[(df['UserId']==19) | (df['UserId']==278418)]






































