import pandas
from sklearn import linear_model

#Multiple regression is like linear regression, but with more than one independent value, meaning that we try to predict a value based on two or more variables.

df = pandas.read_csv("../data_files/cars.csv")

#Then make a list of the independent values and call this variable X.
# Put the dependent values in a variable called y.

X = df[['Weight', 'Volume']]
y = df['CO2']

# LinearRegression() method to create a linear regression object.
# Fit() that takes the independent and dependent values as parameters and fills the regression object with data that describes the relationship:

regr = linear_model.LinearRegression()
regr.fit(X, y)

# predict(x,y) is regression object used to predict the data based on the inputs given.

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300ccm:
predictedCO2 = regr.predict([[2300, 1300]])
print(predictedCO2)

#Coefficient is a factor that describes the relationship with an unknown variable.
#For this sample, we can ask for the coefficient value of weight against CO2, and for volume against CO2
print(regr.coef_)
