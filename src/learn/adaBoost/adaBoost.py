'''
Created on Dec 10, 2017

@author: I309392
'''
import learn.decisionTree.decisionTreeStump as dts
import numpy as np
import math

def __calFalseRate(labelOri, labelResult):
    labelSize = len(labelOri)
    falseSize = 0
    if labelSize == len(labelResult):
        for i in range(labelSize):
            if labelOri[i] != labelResult[i]:
                falseSize += 1
        if falseSize == 0:
            return 0.000001
        else:
            return float(falseSize / labelSize)
    else:
        return -1
    
def __calAlpha(fRate):
    alpha = 1/2 * math.log((1-fRate) / fRate)
    return alpha
    
def adaBoostProcessor(dataSet, labelSet):
    m,n = np.shape((dataSet))
    labelMatrix = np.mat(labelSet).T
    D = np.mat(np.ones((m, 1)))
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
        for lineNo in range(np.shape(D)[0]):
            if bestDividVector[lineNo] == labelMatrix[lineNo]:
                D[lineNo] = D[lineNo] * (math.e**(-alpha)) / sumD
            else:
                D[lineNo] = D[lineNo] * (math.e**(alpha)) / sumD
        #D[np.mat(bestDividVector) == labelMatrix] = D[np.mat(bestDividVector) == labelMatrix] * (math.e**(-alpha)) / sumD
        #D[np.mat(bestDividVector) != labelMatrix] = D[np.mat(bestDividVector) != labelMatrix] * (math.e**(alpha)) / sumD
        result.append(bestStump)
        if fRate - 0.0 < 0.0001:
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

if __name__ == '__main__':
    dataSet = [[0.1, 0.2],[0.2, 0.1],[0.3, 0.1],[0.2, 0.3],[0.7, 0.8],[0.9, 0.6],[0.8, 0.8],[0.7, 0.7]]
    labelSet = [1,1,1,1,-1,-1,-1,-1]
    result = adaBoostClassify(dataSet, labelSet, [[0.1, 0.1]])
    print(result)
        
    