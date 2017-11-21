'''
Created on Nov 15, 2017

@author: Lin
'''
import matplotlib.pyplot as plt
import numpy as np

# input:
#     weight: matrix, ([[],[],...])
#     dataMatrix: array, [[],[],[]...], 
#     labelVector: array, []
# output:
#     a map
def regressDrawer(weight, dataMatrix, labelVector):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dataMatrix = np.array(dataMatrix).transpose()
    redSet = []
    greenSet = []
    for i in range(dataMatrix.shape[1]):
        if labelVector[i] - 0 < 0.000001:
            redSet.append([dataMatrix[0][i], dataMatrix[1][i]])
        else:
            greenSet.append([dataMatrix[0][i], dataMatrix[1][i]])
    redSet = np.mat(redSet).transpose().tolist()
    greenSet = np.mat(greenSet).transpose().tolist()
    ax.scatter(redSet[0], redSet[1], s=30, c='red', marker='s')
    ax.scatter(greenSet[0], greenSet[1], s=30, c='green')
    # draw the line: 0 = w0 + w1*x1 + w2*x2 => w2 = (w1*x1 + w0) / -x2
    x1 = np.arange(0.0, 1.0, 0.1)
    x2 = (-weight[0] - weight[1]*x1) / weight[2]
    x2 = np.array(x2)[0]
    ax.plot(x1, x2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()