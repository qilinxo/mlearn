'''
Created on Dec 8, 2017

@author: Lin
'''
import numpy as np
from math import inf

def falseRate(labelOri, labelResult):
    labelSize = len(labelOri)
    trueSize = 0
    falseSize = 0
    if labelSize == len(labelResult):
        for i in range(labelSize):
            if labelOri[i] == labelResult[i]:
                trueSize += 1
            else:
                falseSize += 1
        return float(trueSize / falseSize)
    else:
        return -1
    
def chooseBestProp(dataSet, labelSet, D):
    dataMatrix = np.array(dataSet)
    labelMatrix = np.array(labelSet).T
    m, n = np.shape(dataMatrix)
    minError = inf
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
                if childT == 'lt':
                    prodictVector[dataMatrix[:, j] <= goStepTo] = -1.0
                else:
                    prodictVector[dataMatrix[:, j] >= goStepTo] = -1.0
                errorVector = np.ones((m, 1))
                errorVector[prodictVector == labelMatrix] = 0.0
                weightError = D.T * errorVector
                if minError > weightError:
                    minError = weightError
                    bestDividVector = prodictVector.copy()
                    bestStump['prop'] = i
                    bestStump['divideVal'] = goStepTo
                    bestStump['childTree'] = childT
    return bestStump, bestDividVector, minError
                    
                
        