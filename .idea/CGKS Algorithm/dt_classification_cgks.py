import pandas
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#Import the dataset
#df = pandas.read_csv(r"data_files\MOCK_DATA_ITERATION_1.csv")
#df = pandas.read_csv(r"data_files\MOCK_DATA_ITERATION_2.csv")

print(df)

#To split the dataset into features and target variable
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

#Split the dataset into 80% training data and 20% of testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 1)

#train the model with the help of DecisionTreeClassifier class of sklearn
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X_train,y_train)

#To make a prediction
y_pred = dtree.predict(X_test)
#print(y_pred)
#print(y_test)

#To obtain the accuracy score, confusion matrix and classification report
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

"""
#Predicting based on data given
print(dtree.predict([[20, 50, 7, 1]]))
print(dtree.predict([[40, 10, 3, 1]]))
print("[1] means 'GO'")
print("[0] means 'NO'")
"""
