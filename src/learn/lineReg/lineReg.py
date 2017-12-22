'''
Created on Dec 15, 2017

@author: I309392
'''
import numpy as np

def ols(dataSet, labelSet):
    dataMatrix = np.mat(dataSet)
    labelMatrix = np.mat(labelSet)
    oneCol = np.ones((np.shape(dataMatrix)[0], 1))
    dataMatrix = np.mat(np.c_[dataMatrix, oneCol])
    xtx = np.dot(dataMatrix.T, dataMatrix)
    w = np.dot(np.dot(xtx*-1, dataMatrix.T), labelMatrix.T)
    return w

def ridgeReg(dataSet, labelSet, alpha):
    dataMatrix = np.mat(dataSet)
    labelMatrix = np.mat(labelSet)
    oneCol = np.ones((np.shape(dataMatrix)[0], 1))
    dataMatrix = np.mat(np.c_[dataMatrix, oneCol])
    n = np.shape(dataMatrix)[1]
    xtx = np.dot(dataMatrix.T, dataMatrix)
    aeye = np.eye(n,n)
    xtx = xtx + aeye
    w = np.dot(np.dot(xtx*-1, dataMatrix.T), labelMatrix.T)
    return w

def eat(dataSet, labelSet, stepSize, iterTimes):
    dataMatrix = np.mat(dataSet)
    labelMatrix = np.mat(labelSet)
    oneCol = np.ones((np.shape(dataMatrix)[0], 1))
    dataMatrix = np.c_[dataMatrix, oneCol]
    m, n = np.shape(dataMatrix)
    w = np.ones((1, n))
    minCost = np.inf
    for time in range(iterTimes):
        for i in range(n):
            for j in [-1, 1]:
                temp = w.copy()
                temp[:, i] = temp[:, i] + ( j * stepSize) 
                xw = np.dot(dataMatrix, temp.T)
                tempCost = labelMatrix.T - xw
                tempCost = np.square(tempCost)
                tempCost = np.sum(tempCost)
                if tempCost < minCost:
                    minCost = tempCost
                    w = temp 
    return w
    
if __name__ == '__main__':
    w = eat([[0.1],[0.2], [0.3], [0.4],[0.5],[0.6],[0.7]], [2,3,4,5,6,7,8], 0.01, 10000)
    w1 = ols([[0.1],[0.2], [0.3], [0.4],[0.5],[0.6],[0.7]], [2,3,4,5,6,7,8])
    w2 = ridgeReg([[0.1],[0.2], [0.3], [0.4],[0.5],[0.6],[0.7]], [2,3,4,5,6,7,8], 1)
    print(w, w1, w2)
    
    