import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import export_graphviz

dataset = pd.read_csv(r"data_files\FINAL_DATASET.csv")
testset = pd.read_csv(r"data_files\KM4U.csv")

dataset.head()
print(dataset)

#Data Preprocessing
X_train = dataset.iloc[:, :-1].values
y_train = dataset.iloc[:, -1].values
X_test = testset.iloc[:, :-1].values
y_test = testset.iloc[:, -1].values
features = ['Context','Acceptance','Accuracy']

#train the model with the help of RandomForestClassifier class of sklearn
#classifier = RandomForestClassifier(n_estimators = 100)
classifier = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0) #Forests of extremely randomized trees
classifier.fit(X_train, y_train)
classifier_limited = classifier.estimators_[5]

##To make a prediction
y_pred = classifier.predict(X_test)
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

export_graphviz(classifier_limited,out_file='trees.dot',
                feature_names = features,
                rounded = True, proportion = False,
                precision = 4, filled = True)
from subprocess import call
call(['dot', '-Tpng', 'trees.dot', '-o', 'images\ rf_classify.png', '-Gdpi=600'])
from IPython.display import Image
Image(filename = 'images\ rf_classify.png')