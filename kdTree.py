import numpy as np
from math import sqrt
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# sklearn

X = [[2, 3], [5, 4], [8, 1], [4, 7], [7, 2], [9, 6]]
y = [1, 0, 0, 0, 0, 0]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y)

print(neigh.predict([[3, 4.5]]))
print(neigh.predict_proba([[3, 4.5]]))

train = np.array([[2, 3, 1], [5, 4, 0], [8, 1, 0], [4, 7, 0], [7, 2, 0], [9, 6, 0]])
for i, arr in enumerate(train):
    train[i] = np.array(arr)
test = np.array([[3, 4.5, 0]])


class Node:
    def __init__(self, data, depth = 0, lchild = None, rchild = None):
        self.data = data
        self.depth = depth
        self.lchild = lchild
        self.rchild = rchild


class KdTree:
    def __init__(self):
        self.KdTree = None
        self.n = 0
        self.nearest = None

    def build(self, dataset, depth=0):
        if len(dataset) > 0:
            m, n = np.shape(dataset)
            self.n = n-1
            axis = depth % self.n
            mid = int(m / 2)
            datasetcopy = sorted(dataset, key = lambda x: x[axis])
            node = Node(datasetcopy[mid], depth)

            if depth == 0:
                self.KdTree = node

            node.lchild = self.build(datasetcopy[ : mid], depth + 1)
            node.rchild = self.build(datasetcopy[mid+1 : ], depth + 1)

            return node
        return None

    def search(self, x, count = 1):
        nearest = []
        for i in range(count):
            nearest.append([-1, None])
        self.nearest = np.array(nearest)

        def recurve(node):
            if node is not None:
                axis = node.depth % self.n
                daxis = x[axis] - node.data[axis]
                if daxis < 0:
                    recurve(node.lchild)
                else:
                    recurve(node.rchild)
                dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(x, node.data)))
                for i, d in enumerate(self.nearest):
                    if d[0] < 0 or dist < d[0]:
                        self.nearest = np.insert(self.nearest, i, [dist, node], axis = 0)
                        self.nearest = self.nearest[:-1]
                        break

                n = list(self.nearest[:, 0]).count(-1)
                if self.nearest[-n-1, 0] > abs(daxis):
                    if daxis < 0:
                        recurve(node.rchild)
                    else:
                        recurve(node.lchild)
        recurve(self.KdTree)

        knn = self.nearest[: 1]
        belong = []
        for i in knn:
            belong.append(i[-1].data[-1])
        b = max(set(belong), key=belong.count)

        return self.nearest, b


kdt = KdTree()
kdt.build(train)

score = 0
for x in test:
    print(x)
    near, belong = kdt.search(x[:-1], 1)
    if belong == x[-1]:
        score += 1
    print('test: ')
    print(x, 'predict:', belong)
    print('nearest:')

    for n in near:
        print(n[1].data, 'dist:', n[0])

