'''
Created on Dec 15, 2017

@author: I309392
'''
import numpy as np

def lineRg(dataSet, labelSet):
    dataMatrix = np.mat(dataSet)
    labelMatrix = np.mat(labelSet)
    oneCol = np.ones((dataMatrix.shape()[0], 1))
    dataMatrix = np.mat(np.c_([dataMatrix, oneCol]))
    xtx = np.dot(dataMatrix.T, dataMatrix)
    w = np.dot(np.dot(xtx*-1, dataMatrix.T), labelMatrix)
    return w

    