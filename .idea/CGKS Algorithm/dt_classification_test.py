import pandas
import numpy as np
import sys
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

def annot_max(x, y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    #xmax = format(str(x[np.argmax(y)]))
    #ymax = format(str(y.max()))
    text= "max={:.3f}".format(ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="left", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.75,0.75), **kw)


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

plt.savefig('fig1.png')
plt.show()