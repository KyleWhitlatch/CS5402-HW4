from sklearn.tree import *
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import csv


def main():
    X = []
    Y = []
    dtreeaccuracy = []
    bestknnaccuracy = []
    testing = []
    with open('testing.csv') as file:
        csvread = csv.reader(file, delimiter=',')
        for row in csvread:
            testing.append([row[0], row[1], row[2], row[3]])
    files = ['training.csv', 'training2.csv', 'training4.csv', 'training4.csv']
    for i,f in enumerate(files):
        with open(f) as file:
            csvread = csv.reader(file, delimiter=',')
            for row in csvread:
                X.append([row[0], row[1], row[2], row[3]])
                Y.append(row[4])
        dtreeaccuracy.append(decisiontree(X,Y,i,testing))
        bestknnaccuracy.append(knn(X,Y,i,testing))
        X = []
        Y = []
    print('Decision Tree Accuracies: ')
    print(dtreeaccuracy)
    print('\n\n')
    print('KNN Accuracy using Best K: ')
    print(bestknnaccuracy)


def decisiontree(X, Y, iteration, testing):
    dtree = DecisionTreeClassifier()
    X = np.array(X)
    Y = np.array(Y)
    dtree.fit(X, Y)
    export_graphviz(dtree,out_file=('dTreeIris.dot' + str(iteration)),class_names=list(set(Y)))
    return dtree.score(testing, Y)


def knn(X,Y,iteration,testing):
    accuracies = []
    for x in range(1,30):
        k = KNeighborsClassifier(n_neighbors=x)
        k.fit(X, Y)
        accuracies.append(k.score(testing, Y))
    print('Accuracies of 1 - 29 K Neighbors for iteration ' + str(iteration) + ':')
    print(accuracies)
    return [(accuracies.index(max(accuracies)) + 1), max(accuracies)]


if __name__ == '__main__':
    main()
