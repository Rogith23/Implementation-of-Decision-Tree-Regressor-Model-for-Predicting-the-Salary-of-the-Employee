# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas

2.Calculate the null values present in the dataset and apply label encoder.

3.Determine test and training data set and apply decison tree regression in dataset.

4.calculate Mean square error,data prediction and r2.  

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Rogith J
RegisterNumber: 212224040280

import pandas as pd
df=pd.read_csv("Salary.csv")
df.head()
df.info()
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"]=le.fit_transform(df["Position"])
df.head()
x=df[["Position","Level"]]
x.head()
y=df["Salary"]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
print("NAME:Rogith J")
print("REG NO: 212224040280")
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
*/
```

## Output:

data.head():

<img width="396" height="239" alt="Screenshot 2025-09-22 134207" src="https://github.com/user-attachments/assets/61cfe8a3-a62c-40e2-ba58-343b1a5c1e66" />

data.info():

<img width="427" height="228" alt="Screenshot 2025-09-22 134313" src="https://github.com/user-attachments/assets/5e215a2c-c884-4d8d-a0c7-950974c6eb62" />

data.isnull().sum():

<img width="328" height="104" alt="Screenshot 2025-09-22 134357" src="https://github.com/user-attachments/assets/869e92ee-1bde-4e81-bc0a-952c692f6875" />

data.head() for salary:

<img width="462" height="227" alt="Screenshot 2025-09-22 134504" src="https://github.com/user-attachments/assets/74e82496-1ef9-4214-808d-5e2c29352eba" />

MSE value:

<img width="416" height="108" alt="Screenshot 2025-09-22 134606" src="https://github.com/user-attachments/assets/05d34745-5012-4c90-a65c-10c3d96194b5" />

r2 value:

<img width="565" height="48" alt="Screenshot 2025-09-22 134644" src="https://github.com/user-attachments/assets/50a5635e-4527-4b2f-a58c-5aa372192e53" />

data prediction:

<img width="394" height="36" alt="Screenshot 2025-09-22 134752" src="https://github.com/user-attachments/assets/b651e817-07f5-4ee3-9ebc-94bb75ad64c8" />

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
