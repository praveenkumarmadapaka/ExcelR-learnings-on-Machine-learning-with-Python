# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 09:28:31 2022

@author: Asher
"""

#step1 : Import the data

import pandas as pd
df = pd.read_csv("Zoo.csv")

df.shape
list(df)

from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df["animal name"] =LE.fit_transform(df["animal name"])

# step2 : split as x and y

X = df.iloc[:,:17]
Y = df.iloc[:,17]

# step3 : Applying the standardization
from sklearn.preprocessing import StandardScaler  
SS = StandardScaler()
X_scale = SS.fit_transform(X)

# step4 :  Train and test split
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test  = train_test_split(X_scale,Y, test_size=0.3)

# step5: model fitting
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=9, p=2)

KNN.fit(X_train,Y_train)

Y_pred_train = KNN.predict(X_train)
Y_pred_test =  KNN.predict(X_test)


# step6 : metrics
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,Y_pred_train)
print("Accuracy score for Training data:", ac1.round(2))
ac2 = accuracy_score(Y_test,Y_pred_test)
print("Accuracy score for Test data:", ac2.round(2))



#k-fold method
from sklearn.model_selection import cross_val_score,KFold

kf=KFold(n_splits=4)
KNN = KNeighborsClassifier(n_neighbors=9, p=2,)
results=cross_val_score(KNN,X_scale,Y,cv=kf,scoring="accuracy")
results.mean()


# after doing kfold accuracy is giviving 88%