# Artificial Neural Networks

###
# Part 1 - Data Preprocessing
###

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')

# Selecting the Features
X = dataset.iloc[:, 3:13].values

# Selecting the Independent Variable Vector
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Normalize Country Feature
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Normalize Gender Feature
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Create dummy variables for categorical features
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# To fix the dummy variable trap, by removing the first column
X = X[:, 1:]

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

###
# Part 2 - Building the Artificial Neural Network
###

# Importing the Keras libraries and required packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
classifier.add(Dropout(p = 0.1))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN, applying Stochastic Gradient Descent
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 5)

###
# Part 3 - Making the Predictions and Evaluating the Model
###

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Classifying the results as either true/false
y_pred = (y_pred > 0.5)

# Predict a single new observation
"""
Use our ANN model to predict if the customer with the following informations will leave the bank: 

Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000

So should we say goodbye to that customer ?
"""
new_obsrv = sc.transform(np.array([[0.0, 0.0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
new_pred = classifier.predict(new_obsrv)
new_pred = (new_pred > 0.5)

"""
Since the value of new_pred is False, the prediction is that the customer will not leave the bank.
"""

###
# Part 4 - Evaluating, Improving and Tuning the ANN
###

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

# The classifier must be a function in order to use KerasClassifier
def build_classifier():
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    
    # Adding the second hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    
    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    # Compiling the ANN, applying Stochastic Gradient Descent
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier

# Use the Keras Classifier with the same parameters as our original classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 5)

# Apply K-Fold Cross Validation
k_fold_accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

# Calculating the Mean and Variance of Accuracies
mean = k_fold_accuracies.mean()
variance = k_fold_accuracies.std()

# Improving the ANN

# Dropout Regularization to Reduce Overfitting, if needed
# aka, Remove Outliers

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier

# Use GridSearchCV instead to tune individual parameters
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

# The classifier must be a function in order to use KerasClassifier
def build_classifier(optimizer):
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    
    # Adding the second hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    
    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    # Compiling the ANN, applying Stochastic Gradient Descent
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier

# Use the Keras Classifier with the same parameters as our original classifier
classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [10, 10],
              'nb_epoch': [5, 5],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5)

grid_search = grid_search.fit(X = X_train, y = y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)