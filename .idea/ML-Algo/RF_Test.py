import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier

digits = datasets.load_digits()
x,y = digits.data, digits.target
series = [10,25,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,3000,4000,5000,6000]
RF = RandomForestClassifier(random_state=101)
train_scores, test_scores = validation_curve(RF,x,y,'n_estimators',param_range=series, cv=10, scoring='accuracy',n_jobs=-1)

plt.figure()
plt.plot(series, np.mean(test_scores,axis=1), '-o')
plt.xlabel('number of trees')
plt.ylabel('accuracy')
plt.grid()
plt.show()