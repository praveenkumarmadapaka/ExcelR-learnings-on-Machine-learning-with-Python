

#hierarchial clustering
import pandas as pd
df = pd.read_csv("crime_data.csv")
df.shape
list(df)

X = df.iloc[:,1:]
X.shape

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_scale = ss.fit_transform(X)


import matplotlib.pyplot as plt
plt.scatter(X.iloc[:,3],X.iloc[:,1])
plt.show()

#dendrogram construction
import scipy.cluster.hierarchy as shc
dend = shc.dendrogram(shc.linkage(X_scale,method="single"))
dend = shc.dendrogram(shc.linkage(X_scale,method='complete'))

from sklearn.cluster import AgglomerativeClustering
#cluster = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="single")
cluster = AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="complete")
Y = cluster.fit_predict(X_scale)
Y_new = pd.DataFrame(Y)
Y_new.value_counts()
pd.concat([df,Y_new],axis=1)


#kmeans
X = df.iloc[:,1:]
X.shape

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_scale = ss.fit_transform(X)

kint_list=[]
for i in range(2,11):
    km = KMeans(n_clusters=i,n_init=20)
    km.fit_predict(X_scale)
    kint_list.append(km.inertia_)
print(kint_list)

import matplotlib.pyplot as plt
plt.scatter(range(2,11),kint_list)
plt.plot(range(2,11),kint_list)
plt.show()
    
from sklearn.cluster import KMeans
km = KMeans(n_clusters=4,n_init=20)
Y = km.fit_predict(X_scale)
Y = pd.DataFrame(Y)
Y.value_counts()
km.inertia_

#DBSCAN
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=1, min_samples=3) #eps = epsalon = radius
dbscan.fit(X_scale)
y=dbscan.labels_
Y=pd.DataFrame(y)
Y.value_counts()



# From above analysis, kmeans forming good number of clusters(4),with good number of samples.
