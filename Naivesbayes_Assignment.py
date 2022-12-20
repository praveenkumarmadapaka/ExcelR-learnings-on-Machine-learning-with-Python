# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 12:48:24 2022

@author: Asher
"""
#Step-1 Import the dataset1
import pandas as pd
df=pd.read_csv("SalaryData_Train.csv")
df.dtypes

#Step-2 Label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

df["workclass"] = LE.fit_transform(df["workclass"])
df["education"] = LE.fit_transform(df["education"])
df["educationno"] = LE.fit_transform(df["educationno"])
df["maritalstatus"] = LE.fit_transform(df["maritalstatus"])
df["occupation"] = LE.fit_transform(df["occupation"])
df["relationship"] = LE.fit_transform(df["relationship"])
df["race"] = LE.fit_transform(df["race"])
df["sex"] = LE.fit_transform(df["sex"])
df["native"] = LE.fit_transform(df["native"])

X_train=df.iloc[:,0:13]
Y_train=df["Salary"]

#Step-3 import the dataset2
df1=pd.read_csv("SalaryData_Test.csv")

#Step-4 label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

df1["workclass"] = LE.fit_transform(df1["workclass"])
df1["education"] = LE.fit_transform(df1["education"])
df1["educationno"] = LE.fit_transform(df1["educationno"])
df1["maritalstatus"] = LE.fit_transform(df1["maritalstatus"])
df1["occupation"] = LE.fit_transform(df1["occupation"])
df1["relationship"] = LE.fit_transform(df1["relationship"])
df1["race"] = LE.fit_transform(df1["race"])
df1["sex"] = LE.fit_transform(df1["sex"])
df1["native"] = LE.fit_transform(df1["native"])

X_test=df1.iloc[:,0:13]
Y_test=df1["Salary"]
    
#Step-5 Model Fitting
from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X_train,Y_train)

Y_pred_train = MNB.predict(X_train)
Y_pred_test = MNB.predict(X_test)

#Step-6 Metrics
from sklearn.metrics import accuracy_score
Training_score = accuracy_score(Y_train,Y_pred_train)
Test_score = accuracy_score(Y_test,Y_pred_test)
print("Training score",Training_score.round(3))
print("Test score",Test_score.round(3))