import pandas as pd
import numpy as np
import sys
import graphviz
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

dataset = pd.read_csv(path, names = headernames)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
features = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']

#Split the dataset into 70% training data and 20% of testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 1)

#train the model with the help of DecisionTreeClassifier class of sklearn
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train,y_train)

#To make a prediction
y_pred = dtree.predict(X_test)
#print(y_pred)
#print(y_test)

#To obtain the accuracy score, confusion matrix and classification report
print("Confusion Matrix:", "\n",metrics.confusion_matrix(y_test, y_pred))

# Model Accuracy: classification report
print("Classification Report:","\n",metrics.classification_report(y_test, y_pred))

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

"""
#Predicting based on data given
print(dtree.predict([[20, 50, 7, 1]]))
print(dtree.predict([[40, 10, 3, 1]]))
print("[1] means 'GO'")
print("[0] means 'NO'")
"""

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, filled=True, rounded=True,
                special_characters=True,feature_names = features)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('decisiontree.png')
Image(graph.create_png())