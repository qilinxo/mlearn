'''
Created on Nov 13, 2017

@author: Lin
'''
import numpy as np
from numpy import tile
import learn.bayes.bayesHelper as bh
import math

# input 
#     dataMatrix: array, [[],[],[]...]
#     labelVector: array, [], the label set
# output
#     propCi1: dict, excample:{0: 0.16666666666666666, 1: 0.16666666666666666}, p(Ci), c stands for the labels
#     propW1: array, [], p(W), w stands for the the value 0 or 1 in data set
#     propWci1: dict, {label1:array([]), label2:array([])...}, p(W|Ci), the probability of w under c
# comment
#     p(W) = p(w1)*p(w2)*p(w3)*...
#     p(w|c) = p(w1|c)*p(w2|c)*p(w3|c)*...
def __trainPwiMatrix(dataMatrix, labelVector):
    propertiesSize = len(dataMatrix[0])
    # each element in dataMatrix plus 1, to deal with the 0 probability
    dataMatrix = dataMatrix + tile(1, (len(labelVector), propertiesSize))
    # the denom also should plus 2
    vectorSize = len(labelVector) * 3
    # p(ci)
    labelDict = {}
    index = 0
    for item in labelVector:
        if item not in labelDict.keys():
            labelDict[item] = []
        labelDict[item].append(dataMatrix[index])
        index += 1
    propCi1 = {}
    for item in labelDict:
        propCi1[item] = len(labelDict[item]) / vectorSize
    # p(w)
    dataMatrix = np.array(dataMatrix)
    propW1 = np.sum(dataMatrix, axis=0)
    propW1 = propW1 / vectorSize
    # p(w|ci)
    propWci1 = {}
    for item in propCi1:
        tempMatrix = np.array(labelDict[item])
        # each item plus 2 to deal with the 0 probability
        cVecterSize = tempMatrix.shape[0] * 3
        tempPropWc = np.sum(tempMatrix, axis = 0) / cVecterSize
        propWci1[item] = tempPropWc
    return propCi1, propW1, propWci1

# input 
#     inputVector: array, [], the item should be judged
#     dataMatrix: array, [[],[],[]...]
#     labelVector: array, [], the label set
# output
#     OutputList[0][0]: the key of dict. the classfication result.
# comment
#     P(c|w) = P(w|c) * P(c) / P(w)
def classifyData(inputVector, dataMatrix, labelVector):
    propCi1, propW1, propWci1 = __trainPwiMatrix(dataMatrix, labelVector)
    resultDict = {}
    for key in propWci1.keys():
        propWc = 1
        propW = 1
        index = 0
        for w in inputVector:
            if w == 1:
                propWc = propWc + math.log(propWci1[key][index], math.e)
                propW = propW + math.log(propW1[index], math.e)
            elif w == 0:
                propWc = propWc + math.log(1 - propWci1[key][index], math.e)
                propW = propW + math.log(1 - propW1[index], math.e)
            index += 1
        p = propWc + propCi1[key] - propW
        resultDict[key] = p
    OutputList = sorted(resultDict.items(), key=lambda d:d[1], reverse = True)
    return OutputList[0][0]
    
if __name__ == '__main__':
    wordsList, labelVector = bh.createDataSet()
    vablist = bh.createVocabList(wordsList)
    print(vablist)
    dataMatrix = bh.wordsToVectoers(vablist, wordsList)
    print(dataMatrix, labelVector)
    propCi, propW, propWci = __trainPwiMatrix(dataMatrix, labelVector)
    print(propCi, propW, propWci)
    print(classifyData([0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], dataMatrix, labelVector))

