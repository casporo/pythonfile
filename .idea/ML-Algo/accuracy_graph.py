import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

train=pd.read_csv('run_train,tag_accuracy_1.csv', sep=',', header=0)
val=pd.read_csv('run_validation,tag_accuracy_1.csv', sep=',', header=0)


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

train_x = train.values[:,1]
train_y = train.values[:,2]
trian_y = savgol_filter(train_y, 51, 3)

val_x = val.values[:,1]
val_y = val.values[:,2]
val_y = savgol_filter(val_y, 51, 3)

fig, ax = plt.subplots()

ax.plot(val_x, val_y, label='validation')
ax.plot(train_x, train_y, label='train')
ax.legend()
plt.title('Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')

annot_max(val_x, val_y)

plt.savefig('fig.png')
plt.show()