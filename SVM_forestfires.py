# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 15:36:43 2022

@author: Asher
"""

# Import the dataset
import pandas as pd
df=pd.read_csv("forestfires.csv")
df.dtypes

# Label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["month"] = LE.fit_transform(df["month"])
df["day"] = LE.fit_transform(df["day"])

# split the X and Y
X=df.iloc[:,:30]
Y=df["size_category"]

# Standardization
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X_scale=SS.fit_transform(X)
X_scale=pd.DataFrame(X_scale)

#splitting the train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_scale,Y,test_size=0.3)

# Model fitting
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=2.0)
#svm = SVC(kernel='poly',degree=3)
#svm = SVC(kernel='rbf', gamma=3)

svm.fit(X_train, Y_train)
Y_pred_train = svm.predict(X_train)
Y_pred_test  = svm.predict(X_test)

# import the metrics
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test, Y_pred_test)
print(cm)
print("Training Accuracy :",accuracy_score(Y_train, Y_pred_train).round(2))
print("Testing Accuracy:",accuracy_score(Y_test, Y_pred_test).round(2))

#resampling methods
#1.validation set approach

import numpy as np
training_accuracy=[]
test_accuracy=[]

for i in range(1,500):
    X_train,X_test,Y_train,Y_test=train_test_split(X_scale,Y,test_size=0.3,random_state=i)
    svm.fit(X_train, Y_train)
    Y_pred_train = svm.predict(X_train)
    Y_pred_test  = svm.predict(X_test)
    training_accuracy.append(accuracy_score(Y_train, Y_pred_train))
    test_accuracy.append(accuracy_score(Y_test, Y_pred_test))

print(training_accuracy)
print(test_accuracy)

np.mean(training_accuracy) #accuracy=93.8%
np.mean(test_accuracy)  #accuracy=92.2%

