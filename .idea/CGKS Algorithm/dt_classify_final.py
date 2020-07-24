import pandas
import numpy as np
import sys
import graphviz
import pydotplus

from sklearn.tree import export_text
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

#Import the dataset
df = pandas.read_csv(r"data_files\FINAL_DATASET.csv")
dt = pandas.read_csv(r"data_files\KM4U.csv")

#To split the dataset into features and target variable
X_train = df.iloc[:, :-1].values
y_train = df.iloc[:, -1].values
X_test = dt.iloc[:, :-1].values
y_test = dt.iloc[:, -1].values
features = ['Context','Acceptance','Accuracy']

#train the model with the help of DecisionTreeClassifier class of sklearn
dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=20, min_samples_split=2, min_samples_leaf=1, max_features='auto', random_state=None)
decision_tree = dtree.fit(X_train,y_train)

#To make a prediction
y_pred = decision_tree.predict(X_test)
#np.set_printoptions(threshold=sys.maxsize)
#print(y_pred)
y_pred = y_pred.tolist()
#print(y_pred)
#print(*y_pred,sep = "\n")

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

r = export_text(dtree, feature_names=features)
print(r)

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, filled=True, rounded=True,
                special_characters=True,feature_names = features,class_names=['Low Quality','Medium Quality', 'High Quality'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('images\dt_classify.png')
Image(graph.create_png())