'''
Created on Nov 14, 2017

@author: I309392
'''
import numpy as np
import mLearnHelper as mh
import regressHelper as helper
import random
import datetime

# input
#     dataMatrix: array, [[],[],[]...], 
#     weight: matrix, ([[],[],...])
# output
#     h: h(x)
def __sigmod(dataMatrix, weight):
    x = dataMatrix * weight
    h = 1.0 / (1 + np.exp(-x))
    return h

# input
#     dataMatrix: array, [[],[],[]...], 
#     lableVector: array, [], 
#     alpha: float
#     times: the loop times
# output
#     weight: matrix, ([[],[],...])
def __gradAscent(dataMatrix, labelVector, alpha, times):
    # set X0 in dataMatrix, each X0 item set 1
    # to avoid reference passing, change to matrix
    dataMatrix = np.mat(dataMatrix)
    dataMatrix = np.column_stack((np.ones((dataMatrix.shape[0], 1)), dataMatrix))
    weight = np.ones((dataMatrix.shape[1], 1))
    y = np.mat(labelVector).transpose()
    for index in range(times):
        h = __sigmod(dataMatrix, weight)
        e = y - h
        weight = weight + alpha * dataMatrix.transpose() * e
    return weight

# input
#     dataMatrix: array, [[],[],[]...], 
#     lableVector: array, [], 
#     times: the loop times
# output
#     weight: matrix, ([[],[],...])
# comment
#     stochastic grad ascent. each loop just calculate one item of dataMatrix, the speed of iteration is much faster. this will reduce the cost of resource
def __stochGradAscent(dataMatrix, labelVector, times):
    dataMatrix = np.mat(dataMatrix)
    dataMatrix = np.column_stack((np.ones((dataMatrix.shape[0], 1)), dataMatrix))
    weight = np.ones((dataMatrix.shape[1], 1))
    for time in range(times):
        indexList = list(range(dataMatrix.shape[0]))
        for i in range(dataMatrix.shape[0]):
            stochIndex = random.choice(indexList)
            alpha = 4/(1.0 + i + time) + 0.01
            h = __sigmod(dataMatrix[stochIndex], weight)
            error = labelVector[stochIndex] - h
            weight = weight + (alpha * error * dataMatrix[stochIndex]).transpose()
            indexList.remove(stochIndex)
    return weight
            
if __name__ == '__main__':
    dataMatrix = np.random.random((100, 2)).tolist()
    labelVector = np.random.random((1, 100)).tolist()[0]
    #print(dataMatrix, labelVector)
    for i in range(100):
        if labelVector[i] > 0.5:
            labelVector[i] = 1
        else:
            labelVector[i] = 0
    print(dataMatrix, labelVector)
    starttime = datetime.datetime.now()
#     weight = __stochGradAscent(dataMatrix, labelVector, 100)
#     endtime = datetime.datetime.now()
#     exeTime = (endtime - starttime).microseconds 
#     print(weight, exeTime)
#     starttime = datetime.datetime.now()
#     weight1 = __gradAscent(dataMatrix, labelVector,0.01, 1000)
#     endtime = datetime.datetime.now()
#     exeTime = (endtime - starttime).microseconds 
#     print(weight1, exeTime)
    exeTime, weight = mh.timeSpend(__stochGradAscent, dataMatrix, labelVector, 100)
    print(weight, exeTime)
    exeTime1, weight1 = mh.timeSpend(__gradAscent, dataMatrix, labelVector, 0.01, 500)
    print(weight1, exeTime1)
    helper.regressDrawer(weight, dataMatrix, labelVector)
    helper.regressDrawer(weight1, dataMatrix, labelVector)

