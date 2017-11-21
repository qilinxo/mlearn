'''
Created on Nov 20, 2017

@author: I309392
'''
import numpy as np
import random
import mLearnHelper as ml
import regressHelper as rh

def __slelectJByRandom(i, m):
    rangeList = list(range(i, m+1))
    j = random.choice(rangeList)
    return j

def svmSmo(dataSet, labelVector, maxIter, C):
    dataMatrix = np.mat(dataSet)
    labelMatrix = np.mat(labelVector).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    a = np.zeros((1, m))
    iter = 0
    while(iter < maxIter):
        aChanged = 0
        L = 0.0
        H = 0.0
        for i in range(m):
            # fxi
            fxi = float(np.multiply(a.T, labelMatrix) * dataMatrix * dataMatrix[i, :]) + b
            ei = fxi - labelVector[i]
            if a[i] < C and a[i] > 0:
                j = __slelectJByRandom(i, m)
                fxj = float(np.multiply(a.T, labelMatrix) * dataMatrix * dataMatrix[j, :]) + b
                ej = fxj - labelVector[j]
                aiOld = a[i].copy()
                ajOld = a[j].copy()
                if labelMatrix[i] != labelMatrix[j]:
                    L = max(0, a[j] - a[i])
                    H = min(C, C + a[j] - a[i])
                else:
                    L = max(0, a[j] - a[i] - C)
                    H = min(C, a[j] - a[i])
            # eta
                eta = dataMatrix[i] * dataMatrix[1] + dataMatrix[j] * dataMatrix[j] + 2 * dataMatrix[i] * dataMatrix[j]
                ajNewTemp = ajOld + labelMatrix[j][0] * (ei - ej) / eta
            # decide a2 new
                ajNew = 0.0
                if ajNewTemp > H:
                    ajNew = H
                elif ajNewTemp >= C and ajNewTemp <= H:
                    ajNew = L
                else:
                    ajNew = ajNewTemp
                a[j] = ajNew
            # a1new
                aiNew = aiOld + labelMatrix[i][0] * labelMatrix[j][0] * (ajOld - a[j])
                a[i] = aiNew
            # b
                b1New = 0 - ei - labelMatrix[i][0] * dataMatrix[i] * dataMatrix[i] * (a[i] - aiOld) - \
                labelMatrix[j][0] * dataMatrix[j] * dataMatrix[i] * (a[j] - ajOld) + b
                b2New = 0 - ej - labelMatrix[i][0] * dataMatrix[i] * dataMatrix[j] * (a[i] - aiOld) - \
                labelMatrix[j][0] * dataMatrix[j] * dataMatrix[2] * (a[j] - ajOld) + b
                if a[i] > 0 and a[i] < C:
                    b = b1New
                elif a[j] > 0 and a[j] < C:
                    b = b2New
                else:
                    b = (b1New + b2New) / 2
                aChanged += 1
        if (aChanged == 0):
            iter += 1
        else:
            iter = 0
    return a, b

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
    exeTime, resultSet = ml.timeSpend(svmSmo, dataMatrix, labelVector, 10, 0.1)
    weight = resultSet[1].extend(resultSet[0])
    rh.regressDrawer(weight, dataMatrix, labelVector)
    