import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from numpy import random
numpy.random.seed(5)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 50, 100) / x

#Split Into Train/Test
# training set should be a random selection of 80% of the original data.
train_x = x[:80]
train_y = y[:80]

#  testing set should be the remaining 20%.
test_x = x[80:]
test_y = y[80:]

"""
#Display the Data Set
plt.scatter(x, y)
plt.show()
"""

"""
#Display the Training Set
plt.scatter(train_x, train_y)
plt.show()

"""
"""
#Display the Test Set
plt.scatter(test_x,test_y)
plt.show()
"""
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

""""
#Draw a polynomial regression + training set
myline = numpy.linspace(0, 6, 100)
plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()
"""

# Calculate the R2(square) Training Set
r2_training = r2_score(train_y, mymodel(train_x))
print("R2 Training Set is:",r2_training)

# Calculate the R2(square) Test Set
r2_test = r2_score(test_y, mymodel(test_x))
print("R2 Test Set is:",r2_test)

for i in range (1500):
    if i % 100 == 0:
        x=random.randint(100)
        print(mymodel(x))


