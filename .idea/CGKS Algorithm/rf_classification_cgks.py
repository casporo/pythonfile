import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import export_graphviz
from scipy.signal import savgol_filter

#dataset = pd.read_csv(r"data_files\TRAINING_DATA_DATASET_A.csv")
#dataset = pd.read_csv(r"data_files\TRAINING_DATA_DATASET_B.csv")
#dataset = pd.read_csv(r"data_files\TRAINING_DATA_DATASET_C.csv")
#dataset = pd.read_csv(r"data_files\TRAINING_DATA_DATASET_D.csv")
#dataset = pd.read_csv(r"data_files\TRAINING_DATA_DATASET_E.csv")
#dataset = pd.read_csv(r"data_files\TRAINING_DATA_DATASET_F.csv")
#dataset = pd.read_csv(r"data_files\TRAINING_DATA_DATASET_G.csv")
#dataset = pd.read_csv(r"data_files\TRAINING_DATA_DATASET_H.csv")
#dataset = pd.read_csv(r"data_files\RANDOM_DATA_DATASET_A.csv")
#dataset = pd.read_csv(r"data_files\RANDOM_DATA_DATASET_B.csv")
#dataset = pd.read_csv(r"data_files\RANDOM_DATA_DATASET_C.csv")
#dataset = pd.read_csv(r"data_files\RANDOM_DATA_DATASET_D.csv")
#dataset = pd.read_csv(r"data_files\RANDOM_DATA_DATASET_E.csv")
#dataset = pd.read_csv(r"data_files\RANDOM_DATA_DATASET_F.csv")
#dataset = pd.read_csv(r"data_files\RANDOM_DATA_DATASET_G.csv")
#dataset = pd.read_csv(r"data_files\RANDOM_DATA_DATASET_H.csv")
dataset = pd.read_csv(r"data_files\FINAL_DATASET.csv")
dataset.head()
print(dataset)

#Data Preprocessing
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
features = ['Knowledge_Context','Knowledge_Acceptance','Knowledge_Accuracy']
#print(X)
#print(y)

#divide the data into train and test split. The following code will split the dataset into 80% training data and 20% of testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#train the model with the help of RandomForestClassifier class of sklearn
classifier = RandomForestClassifier(n_estimators = 100)
classifier.fit(X_train, y_train)
classifier_limited = classifier.estimators_[99]
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

export_graphviz(classifier_limited,out_file='tree.dot',
                feature_names = features,
                class_names = ['Low Quality','Medium Quality', 'High Quality'],
                rounded = True, proportion = False,
                precision = 2, filled = True)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'images\oforest.png', '-Gdpi=600'])
from IPython.display import Image
Image(filename = 'images\oforest.png')