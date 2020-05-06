import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#dataset = pd.read_csv(r"data_files\MOCK_DATA_0.csv")
#dataset = pd.read_csv(r"data_files\MOCK_DATA_1.csv")
#dataset = pd.read_csv(r"data_files\MOCK_DATA_2.csv")
#dataset = pd.read_csv(r"data_files\MOCK_DATA_3.csv")
#dataset = pd.read_csv(r"data_files\MOCK_DATA_4.csv")
#dataset = pd.read_csv(r"data_files\MOCK_DATA_5.csv")
dataset = pd.read_csv(r"data_files\MOCK_DATA_6.csv")
dataset.head()
print(dataset)

#Data Preprocessing
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#print(X)
#print(y)

#divide the data into train and test split. The following code will split the dataset into 80% training data and 20% of testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#train the model with the help of RandomForestClassifier class of sklearn
classifier = RandomForestClassifier(n_estimators = 100)
classifier.fit(X_train, y_train)

##To make a prediction
y_pred = classifier.predict(X_test)

#To obtain the accuracy score, confusion matrix and classification report
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)
