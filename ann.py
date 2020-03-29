# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("Churn_Modelling.csv")
X=df.iloc[:,3:13].values   # Excluding RowNumber,CustId,Surname
y=df.iloc[:,13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
# Country column
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)

# Male/Female
# Male/Female
labelencoder_X = LabelEncoder()
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling (its imp in Dl)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Importing Keras Libraries

from tensorflow.python.keras.layers import Dense

from tensorflow.python.keras import Sequential
# Initialising the ANN
classifier=Sequential()

classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu",input_dim=11))
classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu"))
classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))


classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=["accuracy"])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Prediciton
y_pred=classifier.predict(X_test)
y_pred=(y_pred > 0.5)

# Making Confusion matrix
from sklearn.metrics import confusion_matrix
com=confusion_matrix(y_test,y_pred)
print(com)


""""IMPROVING THE ANN USING K-FOLD CROSS VALIDATION """

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier(optimizer='adam'):
        classifier=Sequential()
        classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
        classifier.add(Dense(units=6,kernel_initializer="uniform",activation='relu'))
        classifier.add(Dense(units=1 ,kernel_initializer="uniform",activation='sigmoid'))
        classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
        return classifier


classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)

mean = accuracies.mean()
variance = accuracies.std()

from sklearn.model_selection import GridSearchCV

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs=-1)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_








