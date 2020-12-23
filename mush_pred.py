# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:19:36 2020

@author: Aakash Babu
"""


import pandas as pd
from sklearn.model_selection import train_test_split

# reading the csv file
train_df = pd.read_csv("D:\Studies\Machine Learning\Mushroom Classification\data\mushrooms.csv")

# since we have no missing values in the data and each data are categorial
#  we can replace the label with dummies as follow.

X = train_df.drop('class', axis=1)
y = train_df['class']
X = pd.get_dummies(X)
y = pd.get_dummies(y)
X_train, X_test, y_train, y_test = train_test_split(X, y)


# Random forest Classifier

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1)
model.fit(X_train, y_train)
model.score(X_test, y_test)
print(model.score(X_test,y_test))


'''
# Logistic Regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter = 500000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(accuracy)
'''