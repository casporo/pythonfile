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
#df = pandas.read_csv(r"data_files\TRAINING_DATA_DATASET_B.csv")
#dt = pandas.read_csv(r"data_files\RANDOM_DATA_DATASET_B.csv")
#df = pandas.read_csv(r"data_files\TRAINING_DATA_DATASET_C.csv")
#dt = pandas.read_csv(r"data_files\RANDOM_DATA_DATASET_C.csv")
#df = pandas.read_csv(r"data_files\TRAINING_DATA_DATASET_D.csv")
#dt = pandas.read_csv(r"data_files\RANDOM_DATA_DATASET_D.csv")
#df = pandas.read_csv(r"data_files\TRAINING_DATA_DATASET_E.csv")
#dt = pandas.read_csv(r"data_files\RANDOM_DATA_DATASET_E.csv")
df = pandas.read_csv(r"data_files\TRAINING_DATA_DATASET_F.csv")
dt = pandas.read_csv(r"data_files\RANDOM_DATA_DATASET_F.csv")
#df = pandas.read_csv(r"data_files\TRAINING_DATA_DATASET_G.csv")
#dt = pandas.read_csv(r"data_files\RANDOM_DATA_DATASET_G.csv")
#df = pandas.read_csv(r"data_files\TRAINING_DATA_DATASET_H.csv")
#dt = pandas.read_csv(r"data_files\RANDOM_DATA_DATASET_H.csv")

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

count_0 = 0
count_1 = 0
count_2 = 0

for num in y_pred:
    if num == 0:
        count_0 += 1
    elif num == 1:
        count_1 += 1
    elif num == 2:
        count_2 += 1

print("----Algorithm Classification Results----")
print("Low Quality:",count_0)
print("Medium Quality:",count_1)
print("High Quality:",count_2)