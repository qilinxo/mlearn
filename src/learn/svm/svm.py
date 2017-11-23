'''
Created on Nov 20, 2017

@author: I309392
'''
import numpy as np
import random
import mLearnHelper as ml
import regressHelper as rh

def __slelectJByRandom(i, m):
    rangeList = list(range(i, m))
    j = random.choice(rangeList)
    return j

def svmSmo(dataSet, labelVector, maxIter, C, toler):
    dataMatrix = np.mat(dataSet)
    labelMatrix = np.mat(labelVector)
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
            fxi = float(np.multiply(a, labelMatrix) * (dataMatrix * dataMatrix[i, :].T)) + b
            ei = fxi - labelVector[i]
            if (labelMatrix[0,i]*ei > -toler and a[0][i] < C) or (labelMatrix[0,i]*ei < toler and a[0][i] > 0):
                j = __slelectJByRandom(i, m)
                fxj = float(np.multiply(a, labelMatrix) * (dataMatrix * dataMatrix[j].T)) + b
                ej = fxj - labelVector[j]
                aiOld = a[0,i].copy()
                ajOld = a[0,j].copy()
                if labelMatrix[0,i] != labelMatrix[0,j]:
                    L = max(0, a[0,j] - a[0,i])
                    H = min(C, C + a[0,j] - a[0,i])
                else:
                    L = max(0, a[0,j] - a[0,i] - C)
                    H = min(C, a[0,j] - a[0,i])
            # eta
                eta = dataMatrix[i] * dataMatrix[1].T + dataMatrix[j] * dataMatrix[j].T + 2 * dataMatrix[i] * dataMatrix[j].T
                ajNewTemp = ajOld + labelMatrix[0, j] * (ei - ej) / eta
            # decide a2 new
                ajNew = 0.0
                if ajNewTemp > H:
                    ajNew = H
                elif ajNewTemp >= C and ajNewTemp <= H:
                    ajNew = L
                else:
                    ajNew = ajNewTemp
                a[0,j] = ajNew
            # a1new
                aiNew = aiOld + labelMatrix[0,i] * labelMatrix[0,j].T * (ajOld - a[0,j])
                a[0,i] = aiNew
            # b
                b1New = 0 - ei - labelMatrix[0,i] * dataMatrix[i] * dataMatrix[i].T * (a[0,i] - aiOld) - labelMatrix[0,j] * dataMatrix[j] * dataMatrix[i].T * (a[0,j] - ajOld) + b
                b2New = 0 - ej - labelMatrix[0,i] * dataMatrix[i] * dataMatrix[j].T * (a[0,i] - aiOld) - labelMatrix[0,j] * dataMatrix[j] * dataMatrix[j].T * (a[0,j] - ajOld) + b
                if a[0,i] > 0 and a[0,i] < C:
                    b = b1New
                elif a[0,j] > 0 and a[0,j] < C:
                    b = b2New
                else:
                    b = (b1New + b2New) / 2
                aChanged += 1
        if (aChanged == 0):
            iter += 1
        else:
            iter = 0
    w = np.multiply(a , labelMatrix) * dataMatrix
    return w[0,:].tolist(), b.tolist()

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
    exeTime, resultSet = ml.timeSpend(svmSmo, dataMatrix, labelVector, 10, 0.1, 0.1)
    x = []
    x.append(resultSet[1])
    x.extend(resultSet[0])
    rh.regressDrawer(x, dataMatrix, labelVector)
    