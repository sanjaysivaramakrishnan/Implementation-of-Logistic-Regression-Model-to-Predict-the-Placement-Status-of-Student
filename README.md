# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by  : Sanjay Sivamakrishnan M.
RegisterNumber:  212223240151
```
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
df = pd.read_csv(r'C:\Users\admin\Desktop\Python_jupyter\ML LEARN\intro_machine_learning\data_sets\Placement_Data.csv')
df.head()
df.info()
df.isnull().sum()
df = df.drop(columns = ['salary','sl_no'])
len(df.columns)
df.info()
columns = ['gender','ssc_b', 'hsc_b','hsc_s', 'degree_t', 'workex', 'specialisation','status']
for col in columns:
    df[col] = df[col].astype('category')
for col in columns:
    df[col] = df[col].cat.codes
df.info()
df.head()
#### Date Splitting
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
X
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.2,random_state=42)
print(f'X_train {X_train.shape}')
print(f'y_train {y_train.shape}')
print(f'X_test {X_test.shape}')
print(f'y_test {y_test.shape}')
# from sklearn.preprocessing import StandardScaler
# st = StandardScaler().fit(X_train)
# X_train = st.fit_transform(X_train)
# X_test = st.transform(X_test)
### Model Implementation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model = LogisticRegression(random_state=0,solver='lbfgs',max_iter=10000)
model.fit(X_train,y_train)
y_pred =  model.predict(X_test)
accuracy_score(y_pred,y_test)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

model.predict([[0,0,0,0,0,0,0,0,0,0,0,0]])


```

## Output:
![image](https://github.com/user-attachments/assets/62143ea1-197a-42b3-8606-b38ab5884a47)
![image](https://github.com/user-attachments/assets/ef4f30f3-18d4-4bd5-9b96-aeed6d9962a8)
![image](https://github.com/user-attachments/assets/71d1e6ca-f864-477d-aac3-2392aa54b244)
![image](https://github.com/user-attachments/assets/932af898-1cef-4833-8a43-727a8d7326b7)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
