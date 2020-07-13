import pandas
from sklearn.svm import SVC
from sklearn import datasets
from sklearn import metrics

#Load dataset
dataset = datasets.load_iris()

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2,random_state=109) # 80% training and 20% test

#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = SVC(kernel='linear') # Linear Kernel
#clf = SVC(kernel='rbf') #Gaussian Kernel
#clf = SVC(kernel='sigmoid') #Sigmoid Kernel
#clf = SVC(kernel='poly', degree=8) #Polynomial Kernel
#clf = SVC(kernel='precoumputed') #Precomputed Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Confusion Matrix:", "\n",metrics.confusion_matrix(y_test, y_pred))

print("Classification Report:", "\n",metrics.classification_report(y_test, y_pred))

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
"""
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
"""