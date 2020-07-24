import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"data_files\Gini.csv")
#dataset = pd.read_csv(r"data_files\Entropy.csv")

x= dataset.values[:, 0]
y1 = dataset.values[:, 1]
y2 = dataset.values[:, 2]
y3 = dataset.values[:, 3]
y4 = dataset.values[:, 4]

# plotting line points
plt.plot(x, y1, label = "Result 1", marker='o')
plt.plot(x, y2, label = "Result 2", marker='o')
plt.plot(x, y3, label = "Result 3", marker='o')
plt.plot(x, y4, label = "Result 4", marker='o')

# Set a title, y axis label and x axis label of the current axes.
plt.title('Classification Results with Gini Criterion', fontsize=14)
#plt.title('Classification Results with Entropy Criterion', fontsize=14)
plt.xlabel('Labels', fontsize=14)
plt.ylabel('Numbers', fontsize=14)
# show a legend & grid on the plot
plt.legend()
plt.grid(True)
# Display a figure.
plt.show()