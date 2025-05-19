# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Split the data into training and testing sets.
5. Convert the text data into a numerical representation using CountVectorizer.
6. Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
7. Finally, evaluate the accuracy of the model.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Ashqar Ahamed S.T
RegisterNumber: 212224240018
```
```
import chardet
file='C:\College\SEM 2\Machine Learning\Exp11\spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)
import pandas as pd
data=pd.read_csv(r"C:\College\SEM 2\Machine Learning\Exp11\spam.csv",encoding='Windows-1252')
print(data.head())
print()
data.info()
print(data.isnull().sum())
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print(y_pred)
print()
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
print(accuracy)
```


## Output:
## Data Reslut:
![result](https://github.com/user-attachments/assets/87a35fa7-993d-4dd7-aa75-ce645051a708)

## Data:
![head](https://github.com/user-attachments/assets/bcd252bb-f02d-4f4b-af2d-fe163571a45d)

## Data Info:
![info](https://github.com/user-attachments/assets/d87d83b6-90a4-4879-b0f9-dd9ef325659f)

## Null Data:
![isnum](https://github.com/user-attachments/assets/697b137c-e45c-4d7e-a182-b5dd5c068fbf)

## Prediction:
![pred](https://github.com/user-attachments/assets/eb2b0945-a05b-4c5f-b7e5-a829fc139579)

## Accuracy:
![accuracy](https://github.com/user-attachments/assets/c8fc292d-4278-4a67-9f51-40082b546feb)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
