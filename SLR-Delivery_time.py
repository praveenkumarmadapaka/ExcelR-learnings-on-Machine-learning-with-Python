# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 20:08:19 2022

@author: Asher
"""

# step1 : import the data
import pandas as pd
df = pd.read_csv("delivery_time.csv")
df.shape

# step2 : Data visualization
# scatter plot
df.plot.scatter(x='Sorting Time',y='Delivery Time')
df.corr()


# step3 : split the variables as two parts
X = df[["Sorting Time"]]
Y = df["Delivery Time"]

# Transformations of X
import numpy as np
#x1=np.sqrt(X) #r2_score=70%
#x1=np.log(X) #r2_score=70%
#x1=np.exp(X) #r2_score=36%
#x1=X**3 #r2_score=57%
#x1=(X+2)**2 #r2_score=65%
x1=np.log10(X) #r2_score=70%

# step4 : Model fitting  --> Bo + B1x1
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x1,Y)

LR.intercept_ # Bo or C value 
LR.coef_# B1 value or m value

import numpy as np
t1 = np.array([[11],[12],[13]])
LR.predict(t1)

# step5 : predicted values
Y_pred = LR.predict(x1)
Y

import matplotlib.pyplot as plt
plt.scatter(x=x1,y=Y)
plt.scatter(x=x1,y=Y_pred)
plt.show()

import matplotlib.pyplot as plt
plt.scatter(x1, Y,  color='black')
plt.plot(x1["Sorting Time"], Y_pred, color='red')
plt.show()

# step6 : metrics
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y,Y_pred)
r2=r2_score(Y, Y_pred)
print("mean squared error:", mse.round(2))
print("Root mean squared error:", np.sqrt(mse).round(2))
print("r2 score:", r2.round(2))


#From the above transformations founded 'log10 transformation' is the best model 



























