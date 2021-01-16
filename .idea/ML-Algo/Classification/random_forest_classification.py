import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

from sklearn.model_selection import train_test_split

#fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction.
from sklearn.ensemble import BaggingClassifier
#fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting
from sklearn.ensemble import RandomForestClassifier
#fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting
from sklearn.ensemble import ExtraTreesClassifier
# by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases
from sklearn.ensemble import AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import export_graphviz
from scipy.signal import savgol_filter
from sklearn import datasets

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

dataset = pd.read_csv(path, names = headernames)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
features = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']

#divide the data into train and test split. The following code will split the dataset into 80% training data and 20% of testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#train the model with the help of Bagging Meta-Estimator class of sklearn
#classifier = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5) # Random subsets of 50% of the samples and 50% of the features

#train the model with the help of RandomForestClassifier class of sklearn
#classifier = RandomForestClassifier(n_estimators = 100, max_depth=None,min_samples_split=2, random_state=0) #Forests of randomized trees

#train the model with the help of ExtraTreesClassifier class of sklearn
classifier = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0) #Forests of extremely randomized trees

#train the model with the help of AdaBoost class of sklearn
#classifier = AdaBoostClassifier(n_estimators=100) # The following example shows how to fit an AdaBoost classifier with 100 weak learners

classifier.fit(X_train, y_train)
classifier_limited = classifier.estimators_[5]
##To make a prediction
y_pred = classifier.predict(X_test)

#To obtain the accuracy score, confusion matrix and classification report
print("Confusion Matrix:", "\n",metrics.confusion_matrix(y_test, y_pred))

# Model Accuracy: classification report
print("Classification Report:","\n",metrics.classification_report(y_test, y_pred))

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

export_graphviz(classifier_limited, out_file='trees.dot',
                feature_names = features,
                rounded = True, proportion = False,
                precision = 4, filled = True)
from subprocess import call
call(['dot', '-Tpng', 'trees.dot', '-o', 'forest.png', '-Gdpi=600'])
from IPython.display import Image
Image(filename ='forest.png')