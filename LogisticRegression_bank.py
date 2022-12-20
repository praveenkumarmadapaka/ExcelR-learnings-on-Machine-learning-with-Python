# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 19:43:26 2022

@author: Asher
"""

# step1 : import the data
# import numpy as np
import pandas as pd
df = pd.read_csv("bank-full.csv",sep =';')
df.shape

# step2 : Data visualization
# scatter plot
df.corr()
df.head()

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["job"] = LE.fit_transform(df["job"])
df["marital"] = LE.fit_transform(df["marital"])
df["education"] = LE.fit_transform(df["education"])
df["housing"] = LE.fit_transform(df["housing"])
df["loan"] = LE.fit_transform(df["loan"])
df["contact"] = LE.fit_transform(df["contact"])
df["month"] = LE.fit_transform(df["month"])
df["poutcome"] = LE.fit_transform(df["poutcome"])
df["default"] = LE.fit_transform(df["default"])
df["y"] = LE.fit_transform(df["y"])


df.to_csv("bank.csv")

from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X_scale=SS.fit_transform(df)
X_scale=pd.DataFrame(X_scale)

# step3 : split the variables as two parts
X = df.iloc[:,:16]
Y = df["y"]

# step4 : Model fitting  --> Bo + B1x1
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X,Y)

# step5 : predicted values
Y_pred = LR.predict(X)
Y
# step6 : metrics
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y,Y_pred)
ac = accuracy_score(Y,Y_pred)
print("Accuracy score:", ac.round(2))

from sklearn.metrics import recall_score,f1_score, precision_score
print("Sensitivity score:", recall_score(Y,Y_pred).round(3))
print("f1_score :", f1_score(Y,Y_pred).round(3))
print("precision_score:", precision_score(Y,Y_pred).round(3))

# specificity
TN = cm[0,0]
FP = cm[0,1]

TNR = TN/(TN+FP)
print("Specificity_score:", TNR.round(3))

