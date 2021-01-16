import pandas as pd
import numpy as np
import sys
import graphviz
import pydotplus

from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

dataset = pd.read_csv(path, names = headernames)

d = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
dataset['Class'] = dataset['Class'].map(d)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
features = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width']

#Split the dataset into 70% training data and 20% of testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 1)

#train the model with the help of DecisionTreeClassifier class of sklearn
dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=20, min_samples_split=2, min_samples_leaf=1, max_features='auto', random_state=0)
dtree = dtree.fit(X_train,y_train)

#To make a prediction
y_pred = dtree.predict(X_test)

#To obtain the accuracy score, confusion matrix and classification report
print("Confusion Matrix:", "\n",metrics.confusion_matrix(y_test, y_pred))

# Model Accuracy: classification report
print("Classification Report:","\n",metrics.classification_report(y_test, y_pred))

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

r = export_text(dtree, feature_names=features)
print(r)

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
