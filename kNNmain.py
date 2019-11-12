#!/usr/bin/env python
# coding: utf-8

# Imorting libraries:


import pandas as pd
from math import sqrt


# Loading datasets:


dataTrain = pd.read_csv(r"C:\Users\Lenovo\Documents\irisTrain.csv", header=None)
dataTest = pd.read_csv(r"C:\Users\Lenovo\Documents\irisTest.csv", header=None)


# Calculating single distance:


def Dist(x, *y):
    dist = 0
    for i in range(len(x)-1):
        dist += (x[i] - y[i])**2
    return sqrt(dist)


# Calculating all distances:


def euclideanDist(df, row):
    distances = list()
    distances = df.apply(Dist, axis=1, args=row)
    return distances


# Taking neighbors:


def getNeighbors(train, test, numNei):
    distances = euclideanDist(train, test)
    distances = pd.concat([dataTrain, distances], axis=1, ignore_index=True)
    distances.sort_values([5], inplace=True)
    neighbors = distances.head(numNei).iloc[:, :-1]
    return neighbors


# Predicting class:


def predict_class(train, test, numNei):
    neighbors = getNeighbors(train, test, numNei)
    out_values = list(neighbors[4])
    prediction = max(set(out_values), key = out_values.count)
    return prediction


# Calculating accuracy:


def getAccuracy(test, pred):
    acc = 0
    for i in range(len(test)):
        if test.iloc[i, -1] == pred[i]:
            acc += 1
    return float(acc/len(test))*100.0


# Main program, performing kNN on dataset:


predictions = list()
for _, row in dataTest.iterrows():
    predictions.append(predict_class(dataTrain, list(row), 7))
    print(f"Predicted: {predict_class(dataTrain, list(row), 7)}, Real: {list(row)[-1]}")
getAccuracy(dataTest, predictions)





