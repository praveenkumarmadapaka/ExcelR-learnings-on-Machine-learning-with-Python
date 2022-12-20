   
#== BuyerRatio_Dataset======================================================

#Ho : All proportions are equal
#H1 : All proportions are not equal

import numpy as np

x1 = np.array([50,550])
x1
x2 = np.array([142,351])
x2
x3 = np.array([131,48])
x3
x4 = np.array([70,350]) 
x4

import pandas as pd

x1=pd.DataFrame(x1)
x2=pd.DataFrame(x2)
x3=pd.DataFrame(x3)
x4=pd.DataFrame(x4)

df = pd.concat([x1,x2,x3,x4],axis=1)
df
df.columns=["East","West","North","south"]
df


east_west_crosstab = pd.crosstab(df['East'], df['West'], 
                                      margins=True)
east_west_crosstab
     
north_south_crosstab = pd.crosstab(df['North'], df['south'], 
                                      margins=True)
north_south_crosstab

def check_categorical_dependency(crosstab_table, confidence_interval):
    stat, p, dof, expected = stats.chi2_contingency(crosstab_table)
    print ("Chi-Square Statistic value = {}".format(stat))
    print ("P - Value = {}".format(p))
    alpha = 1.0 - confidence_interval
    if p <= alpha:
        print('H0 is rejected and H1 is accepted')
    else:
	      print('H1 is rejected and H0 is accepted')
    return expected

exp_table_1 = check_categorical_dependency(east_west_crosstab, 0.95)
pd.DataFrame(exp_table_1)
exp_table_2 = check_categorical_dependency(north_south_crosstab, 0.95)
pd.DataFrame(exp_table_2)
#Proved = H1 is rejected and H0 is accepted

#==========Costomer+OrderForm_dataset====================================================

#Ho : All % are equal
#H1 : All % are not equal

import numpy as np

X1 = np.array([29,271])
X1
X2 = np.array([33,267])
x2
X3 = np.array([31,269])
X3
X4 = np.array([20,280]) 
X4

import pandas as pd

X1=pd.DataFrame(X1)
X2=pd.DataFrame(X2)
X3=pd.DataFrame(X3)
X4=pd.DataFrame(X4)

DF = pd.concat([X1,X2,X3,X4],axis=1)
DF
DF.columns=["Phillipines","Indonesia","Malta","India"]
DF

Phillipines_Indonesia_crosstab = pd.crosstab(DF['Phillipines'], DF['Indonesia'], 
                                      margins=True)
Phillipines_Indonesia_crosstab
     
Malta_India_crosstab = pd.crosstab(DF['Malta'], DF['India'], 
                                      margins=True)
Malta_India_crosstab

def check_categorical_dependency(crosstab_table, confidence_interval):
    stat, p, dof, expected = stats.chi2_contingency(crosstab_table)
    print ("Chi-Square Statistic value = {}".format(stat))
    print ("P - Value = {}".format(p))
    alpha = 1.0 - confidence_interval
    if p <= alpha:
        print('H0 is rejected and H1 is accepted')
    else:
	      print('H1 is rejected and H0 is accepted')
    return expected

exp_table_3 = check_categorical_dependency(Phillipines_Indonesia_crosstab, 0.95)
pd.DataFrame(exp_table_3)
exp_table_4 = check_categorical_dependency(Malta_India_crosstab, 0.95)
pd.DataFrame(exp_table_4)   

#Proved = H1 is rejected and H0 is accepted

#==============================================================================================================























    