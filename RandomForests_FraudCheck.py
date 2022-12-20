
# Import the data
import pandas as pd
df = pd.read_csv("Fraud_check.csv")

# EDA
new_Y=[]
for i in df["Taxable.Income"]:
    if i<=30000:
        new_Y.append("Risky")
    else:
        new_Y.append("Good")
new_Y = pd.DataFrame(new_Y)
new_df = pd.concat([df,new_Y],axis=1)
new_df.drop(new_df.columns[[2]],axis=1,inplace=True)
new_df

# label encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
new_df["Undergrad"] = LE.fit_transform(new_df["Undergrad"])
new_df["Marital.Status"] = LE.fit_transform(new_df["Marital.Status"])
new_df["Urban"] = LE.fit_transform(new_df["Urban"])
new_df

# Seperate the X and Y
X = new_df.iloc[:,0:4]
Y = new_Y

# Standardisation
from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
X = SS.fit_transform(X)

# Splitting the train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.3)

# Model fitting
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=(4)) 
dt.fit(X_train, Y_train)
Y_pred_train = dt.predict(X_train) 
Y_pred_test = dt.predict(X_test) 

# Metrics
from sklearn.metrics import accuracy_score
print("Training Accuracy: ",accuracy_score(Y_train,Y_pred_train).round(2))
print("Test Accuracy: ",accuracy_score(Y_test,Y_pred_test).round(2))

# Ensemble methods
from sklearn.ensemble import RandomForestClassifier
RFR = RandomForestClassifier(n_estimators=150,max_samples=0.8,max_features=0.4,max_depth=(5)) 
RFR.fit(X_train, Y_train)
Y_pred_train = RFR.predict(X_train) 
Y_pred_test = RFR.predict(X_test) 

# Metrics
from sklearn.metrics import accuracy_score
print("Training accuracy: ",accuracy_score(Y_train,Y_pred_train).round(2))
print("Test accuracy: ",accuracy_score(Y_test,Y_pred_test).round(2))
