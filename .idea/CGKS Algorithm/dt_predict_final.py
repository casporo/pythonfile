import pandas
import numpy as np
import sys

from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#Import the dataset
#df = pandas.read_csv(r"data_files\TRAINING_DATA_DATASET_A.csv")
#dt = pandas.read_csv(r"data_files\RANDOM_DATA_DATASET_A.csv")

#To split the dataset into features and target variable
X_train = df.iloc[:, :-1].values
y_train = df.iloc[:, -1].values
X_test = dt.iloc[:, :-1].values
y_test = dt.iloc[:, -1].values

#train the model with the help of DecisionTreeClassifier class of sklearn
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train,y_train)

#To make a prediction
y_pred = dtree.predict(X_test)
#np.set_printoptions(threshold=sys.maxsize)
#print(y_pred)
y_pred = y_pred.tolist()
print(y_pred)


"""
print("[2] means 'High'")
print("[1] means 'Low'")
"""
print("[2] means 'High'")
print("[1] means 'Medium'")
print("[0] means 'Low'")

"""
#To obtain the accuracy score, confusion matrix and classification report
result = confusion_matrix(y_train, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_train, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_train,y_pred)
print("Accuracy:",result2)
"""