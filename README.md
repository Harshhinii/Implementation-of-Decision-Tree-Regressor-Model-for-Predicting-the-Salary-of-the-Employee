# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import and Load Data: Import libraries (pandas, sklearn) and load the employee salary dataset.

2.Preprocess Data: Handle missing values, encode categorical data if needed, and define features (X) and target (y).

3.Split Data: Split the dataset into training and testing sets using train_test_split().

4.Train Model: Initialize a DecisionTreeRegressor and train it using the training data.

5.Predict and Evaluate: Predict salary on the test set and evaluate using metrics like Mean Squared Error (MSE) or R² score.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Harshini R
RegisterNumber: 212223220033  
*/
```
```
import pandas as pd
df=pd.read_csv("Salary.csv")
df.head()
```
<img width="823" height="251" alt="image" src="https://github.com/user-attachments/assets/ad12b36c-d28d-4bc4-b40f-233ae9ea9345" />

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Position"]=le.fit_transform(df["Position"])
df.head()
```

<img width="943" height="265" alt="image" src="https://github.com/user-attachments/assets/5936495f-1142-4a71-b697-be99c5f1de51" />

```
x=df[["Position","Level"]]
x.head()
```
<img width="589" height="275" alt="image" src="https://github.com/user-attachments/assets/cfa134f1-e687-40c8-88f9-22881658c14e" />

```
y=df["Salary"]
y.head()
```
<img width="1287" height="173" alt="image" src="https://github.com/user-attachments/assets/636c7602-ca05-4094-bc65-1598a7fa1473" />

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print("Name: Harshini R")
print("RegNo: 212223220033")
print(y_pred)
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import numpy as np
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error:",mse)
print("Root Mean Squared Error:",rmse)
print("Mean Absolute Error:",mae)
print("R2 score:",r2)
dt.predict(pd.DataFrame([[5,6]],columns=["Position","Level"]))
```


## Output:

<img width="1122" height="514" alt="image" src="https://github.com/user-attachments/assets/3a45ac62-1ebd-459e-868d-25af820fc2a2" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
