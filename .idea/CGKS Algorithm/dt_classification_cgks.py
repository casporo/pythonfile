import pandas
import numpy as np
import sys

from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#Import the dataset
#df = pandas.read_csv(r"data_files\TRAINING_DATA_DATASET_A.csv")
#df = pandas.read_csv(r"data_files\TRAINING_DATA_DATASET_B.csv")
#df = pandas.read_csv(r"data_files\TRAINING_DATA_DATASET_C.csv")
#df = pandas.read_csv(r"data_files\TRAINING_DATA_DATASET_D.csv")
#df = pandas.read_csv(r"data_files\TRAINING_DATA_DATASET_E.csv")
#df = pandas.read_csv(r"data_files\TRAINING_DATA_DATASET_F.csv")
#df = pandas.read_csv(r"data_files\TRAINING_DATA_DATASET_G.csv")
#df = pandas.read_csv(r"data_files\TRAINING_DATA_DATASET_H.csv")
#df = pandas.read_csv(r"data_files\RANDOM_DATA_DATASET_A.csv")
#df = pandas.read_csv(r"data_files\RANDOM_DATA_DATASET_B.csv")
#df = pandas.read_csv(r"data_files\RANDOM_DATA_DATASET_C.csv")
#df = pandas.read_csv(r"data_files\RANDOM_DATA_DATASET_D.csv")
#df = pandas.read_csv(r"data_files\RANDOM_DATA_DATASET_E.csv")
#df = pandas.read_csv(r"data_files\RANDOM_DATA_DATASET_F.csv")
#df = pandas.read_csv(r"data_files\RANDOM_DATA_DATASET_G.csv")
#df = pandas.read_csv(r"data_files\RANDOM_DATA_DATASET_H.csv")
print(df)

#To split the dataset into features and target variable
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Split the dataset into 80% training data and 20% of testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 1)

#train the model with the help of DecisionTreeClassifier class of sklearn
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train,y_train)

#To make a prediction
y_pred = dtree.predict(X_test)
#np.set_printoptions(threshold=sys.maxsize)
#print(y_pred)
#print(y_test)

#To obtain the accuracy score, confusion matrix and classification report
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

