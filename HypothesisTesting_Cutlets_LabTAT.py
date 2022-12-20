#====================================================================================================================

#Ho : µ1 = µ2
#H1 : µ1 != µ2

import pandas as pd
df = pd.read_csv("Cutlets.csv")
df.head()

from scipy import stats
Zcal, Pval = stats.ttest_ind(df['Unit A'],df['Unit B'])

alpha = 0.05

if Pval<alpha:
    print("Ho is rejected, H1 is accepted")
else:
    print("Ho is accepted, H1 is rejected")
    
#Proved = Ho is accepted, H1 is rejected

#====================================================================================================================

#Ho : µ1 = µ2 = µ3 = µ4
#H1 : any µ is not equal

import pandas as pd
df1 = pd.read_csv("LabTAT.csv")
df1.head()

import scipy.stats as stats
ftab,pval = stats.f_oneway(df1['Laboratory 1'],df1['Laboratory 2'],df1['Laboratory 3'],df1['Laboratory 4'])
alpha = 0.05

if pval<alpha:
    print("Ho is rejected, H1 is accepted")
else:
    print("Ho is accepted, H1 is rejected")
    
#Proved = Ho is rejected, H1 is accepted

#====================================================================================================================


