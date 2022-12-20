#Import the data
import pandas as pd
df = pd.read_csv("book.csv",encoding=("latin1"))

import numpy as np
df.shape
df.head()

df.sort_values('User.ID')

#number of unique users in the dataset
len(df)
#len(df."User.ID".unique())

df['Book.Rating'].value_counts()
df['Book.Rating'].hist()


#len(df.Book.Title.unique())

#df.Book.Title.value_counts()

user_df = df.pivot(index='User.ID',
                                 columns='Book.Title',
                                 values='Book.Rating')

user_df
user_df.iloc[0]
user_df.iloc[200]
list(user_df)

#Impute those NaNs with 0 values
user_df.fillna(0, inplace=True)
user_df.shape

# from scipy.spatial.distance import cosine correlation
# Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
user_sim = 1 - pairwise_distances(user_df.values,metric='cosine')
user_sim.shape

#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)

#Set the index and column names to user ids 
##user_sim_df.index   = df.User.ID.unique()
##user_sim_df.columns = df.User.ID.unique()

user_sim_df.iloc[0:5, 0:5]

np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:7, 0:7]

#user_sim_df.to_csv("cosin_calc.csv")

#Most Similar Users
user_sim_df.max()

user_sim_df.idxmax(axis=1)[0:10]