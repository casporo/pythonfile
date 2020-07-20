import pandas
import numpy as np
import sys
import graphviz
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as pltimg

from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
from scipy.signal import savgol_filter

def annot_max(x, y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "max={:.3f}".format(ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="left", va="bottom")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.75,0.75), **kw)

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
dtree = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=20, min_samples_split=2, min_samples_leaf=1, max_features='auto', random_state=0)
decision_tree = dtree.fit(X_train,y_train)

#To make a prediction
y_pred = decision_tree.predict(X_test)
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


dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data, filled=True, rounded=True,
                special_characters=True,feature_names = features,class_names=['Low Quality','Medium Quality', 'High Quality'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('images\dt_classify.png')
Image(graph.create_png())

train_x = df.values[:,1]
train_y = df.values[:,2]
train_y = savgol_filter(train_y, 51, 3)

val_x = dt.values[:,1]
val_y = dt.values[:,2]
val_y = savgol_filter(val_y, 51, 3)

fig, ax = plt.subplots()

ax.plot(val_x, val_y, label='validation')
ax.plot(train_x, train_y, label='train')
ax.legend()
plt.title('Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')

annot_max(val_x, val_y)

plt.savefig('images\decision_tree_accuracy.png')
plt.show()