import pandas as pd
import numpy as np
import math
import operator
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import datasets

np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target

y_pred = DBSCAN(eps=0.5, min_samples=5).fit_predict(X)

colors = 'gbycm'
y_pred_color = []
category = []
for pred in y_pred:
    if pred == -1:
        color = 'r'
    else:
        color = colors[pred]
    y_pred_color.append(color)

for type in X:
    if type == 'setosa':
        category.append(0)
    elif type == 'versicolor':
        category.append(1)
    elif type == 'virginica':
        category.append(2)

# print(X)
# print(y_pred)
plt.scatter(y.tolist(), category, c=y_pred_color)
plt.show()


k = 4
distances = {}
for x in range(len(X)):
    distancesForRow = []
    for y in range(len(X)):
        if x != y:
            dist = euclidean_distances(X, y)
            distancesForRow.append(dist)
    distancesForRow.sort(reverse = True)
    distances[x] = distancesForRow[k-1]

newDistances = sorted(distances.items(), key = operator.itemgetter(1))
for x in range(len(newDistances)):
    plt.scatter(x, newDistances[x][1])

plt.show()
