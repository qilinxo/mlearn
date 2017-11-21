'''
Created on Nov 20, 2017

@author: I309392
'''
import numpy as np
def svmSmo(dataSet, labelVector, maxIter, c, toler):
    dataMatrix = np.mat(dataSet)
    labelMatrix = np.mat(labelVector).transpose()
    b = 0
    m, n = np.shape(dataMatrix)
    a = np.zeros((1, m))
    iter = 0
    while(iter < maxIter):
        aChanged = 0
        for i in range(m):
            #fxi
            fxi = float(np.multiply(a.T, labelMatrix) * dataMatrix * dataMatrix[i, :]) + b
            ei = fxi - labelVector[i]
            if a[i] < c and a[i] > 0:
                j = slelectJByRandom(i, m)
                fxj = float(np.multiply(a.T, labelMatrix) * dataMatrix * dataMatrix[j, :]) + b
                ej = fxj - labelVector[j]
                aiOld = a[i].copy()
                ajOld = a[j].copy()
                if labelMatrix[i] != labelMatrix[j]:
                    
            #eta
            #a2
            #a1
            #b
        if (aChanged == 0):
            iter += 1
        else:
            iter = 0
    return a, b