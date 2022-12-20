# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 09:11:06 2022

@author: Asher
"""
#step1 : Import the data

import pandas as pd
df = pd.read_csv("glass.csv")

df.shape
list(df)

# step2 : split as x and y

X = df.iloc[:,0:9]
Y = df["Type"]

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

#Accuracy score for Training data=72%
#Accuracy score for Test data=65%


#============================================================================================================

#resampling methods

#k-fold method
from sklearn.model_selection import cross_val_score,KFold
kf=KFold(n_splits=20)
KNN = KNeighborsClassifier(n_neighbors=4, p=2)
results=cross_val_score(KNN,X_scale,Y,cv=kf,scoring="accuracy")
results.mean()
a = []
b = []

for i in range(2,30):
    for j in range(5,20):
        kf=KFold(n_splits=i)
        a.append(i)
        KNN = KNeighborsClassifier(n_neighbors=j, p=2)
        results_new=cross_val_score(KNN,X_scale,Y,cv=kf,scoring="accuracy")
        b.append(results_new.mean())
df1 = pd.DataFrame()
df1['a'] =  pd.DataFrame(a)
df1['b'] =  pd.DataFrame(b)
df1
max(b)

#While doing kfold the accuracy was maximum 59% only

#================================================================================================================
#validation set approach


training_accuracy=[]
test_accuracy=[]

for i in range(1,500):
    X_train,X_test,Y_train,Y_test=train_test_split(X_scale,Y,test_size=0.3,random_state=i)
    KNN.fit(X_train,Y_train)
    Y_pred_train=KNN.predict(X_train)
    Y_pred_test=KNN.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train, Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test, Y_pred_test))

print(training_accuracy)
print(test_accuracy)

import numpy as np
np.mean(training_accuracy)
np.mean(test_accuracy)

#while doing validation set of approach the training accuracy was 70% and training accuracy was 63%

#===================================================================================================================










