
import sys
import pandas as pd
import matplotlib
import sklearn
import numpy as np
import scipy as sp
import IPython

from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("Keys in iris dataset: \n{}".format(iris_dataset.keys()))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state = 0)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(X_train, y_train)

print('Training accuracy {:.2f}'.format(knn.score(X_train, y_train)))

X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print("Predictrd value: {}".format(prediction))
print('Predicted target name: {}'.format(iris_dataset['target_names'][prediction]))

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))