'''
Created on Dec 26, 2017

@author: I309392
'''
import numpy as np

def chooseBestSplitFeature(dataSet):
    dataMatrix = np.mat(dataSet)
    featrueIndex = -1
    varError = -1
    leftSet = []
    rightSet = []
    for index in dataMatrix:
        for splitValue in dataMatrix[:, index]:
            matLeft, matRight = __binSplit(dataSet, index, splitValue)
            newError = __regError(matLeft) + __regError(matRight)
            if newError < varError:
                varError = newError
                featrueIndex = index
                leftSet = matLeft
                rightSet = matRight
    return featrueIndex, matLeft, matRight

def createTree(dataSet):
    resultTree = {}
    dataMatrix = np.mat(dataSet)
    featrueIndex, matLeft, matRight = chooseBestSplitFeature(dataMatrix)
    temp = []
    if len(matLeft) > 1:
        left = createTree(matLeft)
        temp.append(matLeft)
    if len(matRight) > 1:
        right = createTree(matRight)
        temp.append(matRight)
    if temp != []:
        resultTree[featrueIndex].append(temp)
    return resultTree

def __regError(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]

def __regLeaf(dataSet):
    return np.mean(dataSet[:, -1])

def __binSplit(dataSet, index, splitValue):
    dataMatrix = np.mat(dataSet)
    matLeft = dataMatrix[dataMatrix[:, index] < splitValue, :]
    matRight = dataMatrix[dataMatrix[:, index] >= splitValue, :]
    return matLeft, matRight