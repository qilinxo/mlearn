'''
Created on Dec 8, 2017

@author: Lin
'''
import numpy as np
from math import inf
    
def chooseBestProp(dataSet, labelSet, D):
    dataMatrix = np.array(dataSet)
    labelMatrix = np.array(labelSet).T
    m, n = np.shape(dataMatrix)
    minWeightError = inf
    bestStump = {}
    for i in n:
        steps = 10.0
        minVal = dataMatrix[:, i].min()
        maxVal = dataMatrix[:, i].max()
        rangeVal = maxVal - minVal
        stepSize = rangeVal / steps
        for j in range(steps):
            goStepTo = minVal + j * stepSize
            prodictVector = np.ones((m, 1))
            for childT in ['lt', 'rt']:
                prodictVector = stumpClassify(dataSet, j, goStepTo, childT)
                errorVector = np.ones((m, 1))
                errorVector[prodictVector == labelMatrix] = 0.0
                weightError = D.T * errorVector
                if minWeightError > weightError:
                    minWeightError = weightError
                    bestDividVector = prodictVector.copy()
                    bestStump['prop'] = i
                    bestStump['divideVal'] = goStepTo
                    bestStump['childTree'] = childT
    return bestStump, bestDividVector, minWeightError
                    
def stumpClassify(dataSet, prop, divideVal, childTree):
    dataMatrix = np.mat(dataSet)
    m, n = np.shape(dataMatrix)
    prodictVector = np.ones((m, 1))
    if childTree == 'lt':
        prodictVector[dataMatrix[:, prop] <= divideVal] = -1.0
    else:
        prodictVector[dataMatrix[:, prop] >= divideVal] = -1.0
    return prodictVector