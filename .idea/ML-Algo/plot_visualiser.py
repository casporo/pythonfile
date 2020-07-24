import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import savgol_filter

#dataset = pd.read_csv(r"data_files\cars.csv")

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

dataset = pd.read_csv(path, names = headernames)

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

train_x = dataset.values[:, 4]
train_y = dataset.values[:,2]
train_y = savgol_filter(train_y, window_length=7, polyorder=3)

fig, ax = plt.subplots()

ax.plot(train_x, train_y, label='validation')
ax.legend()
plt.title('Accuracy')
plt.xlabel('Step')
plt.ylabel('Accuracy')

annot_max(train_x, train_y)

plt.savefig('plot_visualiser.png')
plt.show()