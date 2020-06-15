import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#dataset = pd.read_csv(r"data_files\TRAINING_DATA_DATASET_A.csv")
#testset = pd.read_csv(r"data_files\RANDOM_DATA_DATASET_A.csv")
#dataset = pd.read_csv(r"data_files\TRAINING_DATA_DATASET_B.csv")
#testset = pd.read_csv(r"data_files\RANDOM_DATA_DATASET_B.csv")
#dataset = pd.read_csv(r"data_files\TRAINING_DATA_DATASET_C.csv")
#testset = pd.read_csv(r"data_files\RANDOM_DATA_DATASET_C.csv")
#dataset = pd.read_csv(r"data_files\TRAINING_DATA_DATASET_D.csv")
#testset = pd.read_csv(r"data_files\RANDOM_DATA_DATASET_D.csv")
#dataset = pd.read_csv(r"data_files\TRAINING_DATA_DATASET_E.csv")
#testset = pd.read_csv(r"data_files\RANDOM_DATA_DATASET_E.csv")
dataset = pd.read_csv(r"data_files\TRAINING_DATA_DATASET_F.csv")
testset = pd.read_csv(r"data_files\RANDOM_DATA_DATASET_F.csv")
#dataset = pd.read_csv(r"data_files\TRAINING_DATA_DATASET_G.csv")
#testset = pd.read_csv(r"data_files\RANDOM_DATA_DATASET_G.csv")
#dataset = pd.read_csv(r"data_files\TRAINING_DATA_DATASET_H.csv")
#testset = pd.read_csv(r"data_files\RANDOM_DATA_DATASET_H.csv")

dataset.head()
print(dataset)

#Data Preprocessing
X_train = dataset.iloc[:, :-1].values
y_train = dataset.iloc[:, -1].values
X_test = testset.iloc[:, :-1].values
y_test = testset.iloc[:, -1].values

#train the model with the help of RandomForestClassifier class of sklearn
classifier = RandomForestClassifier(n_estimators = 100)
classifier.fit(X_train, y_train)

##To make a prediction
y_pred = classifier.predict(X_test)
#np.set_printoptions(threshold=sys.maxsize)
#print(y_pred)
y_pred = y_pred.tolist()
print(y_pred)

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