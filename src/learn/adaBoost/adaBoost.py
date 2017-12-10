'''
Created on Dec 10, 2017

@author: I309392
'''
import learn.decisionTree.decisionTreeStump as dts
import numpy as np
import math

def __calFalseRate(labelOri, labelResult):
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
    
def __calAlpha(fRate):
    alpha = 1/2 * math.log((1-fRate) / fRate)
    return alpha
    
def adaBoostProcessor(dataSet, labelSet):
    m,n = np.shape((dataSet))
    labelMatrix = np.mat(np.array(labelSet).T)
    D = np.mat(np.ones(m, 1))
    fRate = math.inf
    alpha = math.inf
    bestStump = {}
    bestDividVector = []
    minWeightError = 0.0
    times = 0
    result = []
    while True:
        bestStump, bestDividVector, minWeightError = dts.chooseBestProp(dataSet, labelSet, D)
        sumD = np.sum(D)
        fRate = __calFalseRate(labelSet, bestDividVector)
        alpha = __calAlpha(fRate)
        bestStump['alpha'] = alpha
        D[bestDividVector == labelMatrix] = D[bestDividVector == labelMatrix] * (math.e**(-alpha)) / sumD
        D[bestDividVector != labelMatrix] = D[bestDividVector != labelMatrix] * (math.e**(alpha)) / sumD
        result.append(bestStump)
        if fRate - 0.0 < 0.0000001:
            break
        times += 1
    print(times)
    return result

def adaBoostClassify(dataSet, labelSet, targetSet):
    classifyInfoArray = adaBoostProcessor(dataSet, labelSet)
    dataMatrix = np.mat(dataSet)
    targetMatrix = np.mat(targetSet)
    m,n = np.shape(dataMatrix)
    resultVector = np.zeros((m, 1))
    for dict in classifyInfoArray:
        prodictVector = dts.stumpClassify(dataSet, dict['prop'], dict['divideVal'], dict['childTree'])
        resultVector = resultVector + dict['alpha'] * prodictVector
        print(resultVector)
    return np.sign(resultVector)
        
    