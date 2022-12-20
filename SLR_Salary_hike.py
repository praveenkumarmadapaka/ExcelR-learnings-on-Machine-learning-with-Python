# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:53:42 2022

@author: Asher
"""

# step1 : import the data
import pandas as pd
df = pd.read_csv("Salary_Data.csv")
df.shape

# step2 : Data visualization
# scatter plot
df.plot.scatter(x='YearsExperience',y='Salary')
df.corr()


# step3 : split the variables as two parts
X = df[["YearsExperience"]] #r2_score=96%
Y = df["Salary"]

# Transformations of X
import numpy as np
#x1=np.sqrt(X) #r2_score=93%
#x1=np.log(X) #r2_score=85%
#x1=np.exp(X) #r2_score=47%
#x1=X**3 #r2_score=83%
#x1=(X+2)**2 #r2_score=93%
#x1=np.log10(X) #r2_score=85%

# step4 : Model fitting  --> Bo + B1x1
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X,Y)

LR.intercept_ # Bo or C value 
LR.coef_# B1 value or m value

t1 = np.array([[11],[12],[13]])
LR.predict(t1)

# step5 : predicted values
Y_pred = LR.predict(X)
Y

import matplotlib.pyplot as plt
plt.scatter(x=X,y=Y)
plt.scatter(x=X,y=Y_pred)
plt.show()

import matplotlib.pyplot as plt
plt.scatter(X, Y,  color='black')
plt.plot(X["YearsExperience"], Y_pred, color='red')
plt.show()

# step6 : metrics
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y,Y_pred)
r2=r2_score(Y, Y_pred)
print("mean squared error:", mse.round(2))
print("Root mean squared error:", np.sqrt(mse).round(2))
print("r2 score:", r2.round(2))


#From the above analysis we got higher accuracy without transformations



























