import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"data_files\Dataset_Findings.csv")

x= dataset.values[:, 0]
y = dataset.values[:, 1]

# plotting line points
plt.plot(x, y, marker='o')

# Set a title, y axis label and x axis label of the current axes.
plt.title('Accuracy Results', fontsize=14)
plt.xlabel('Dataset Combination', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
# show a legend & grid on the plot
plt.legend()
plt.grid(True)
# Display a figure.
plt.show()