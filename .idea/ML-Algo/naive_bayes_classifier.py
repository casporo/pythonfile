#Import scikit-learn dataset library
from sklearn import datasets

#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

#Import multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB # Suitable for classification with discrete features (e.g., word counts for text classification).

#Import Complement Naive Bayes model
from sklearn.naive_bayes import ComplementNB #It is particularly suited for imbalanced data sets.

#Import Bernoulli Naive Bayes model
from sklearn.naive_bayes import BernoulliNB # Suitable for discrete data though BernoulliNB is designed for binary/boolean features

#Import Categorical Naive Bayes model
from sklearn.naive_bayes import CategoricalNB #Suitable for classification with discrete features that are categorically distributed

#Load dataset
dataset = datasets.load_iris()

# print the names of the 13 features
print("Features: ", dataset.feature_names)

# print the label type of wine(class_0, class_1, class_2)
print("Labels: ", dataset.target_names)

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3,random_state=109) # 70% training and 30% test

#Create a Gaussian Classifier
#nb = GaussianNB()

#Create a Muntinomial Classifier
nb = MultinomialNB()

#Create a Complement Classifier
#nb = ComplementNB()

#Create a Bernoulli Classifier
#nb = BernoulliNB()

#Create a Categorical Classifier
#nb = CategoricalNB()

#Train the model using the training sets
nb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = nb.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

#To obtain the accuracy score, confusion matrix and classification report
print("Confusion Matrix:", "\n",metrics.confusion_matrix(y_test, y_pred))

# Model Accuracy: classification report
print("Classification Report:","\n",metrics.classification_report(y_test, y_pred))

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))