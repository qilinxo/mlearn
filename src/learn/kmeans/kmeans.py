import numpy as np 
import math 

def __distance(pointA, pointB):
    x = pointA[0] - pointB[0]
    y = pointA[1] - pointB[1]
    dis = abs(x) + abs(y)
    return dis 

def __chooseClusterCenter(dataMatrix):
    m, n = np.shape(dataMatrix)
    result = np.zeros((1,n))
    for point in dataMatrix:
        result += point 
    result = result / m 
    return result 

def __chooseInitCenter(dataSet, k):
    m, n = np.shape(dataSet)
    i = np.random.random() * m
    initCenter = dataSet[int(i)]
    centerSet = []
    centerSet.append(initCenter)
    for i in range(1, k):
        dis = 0
        targetCenter = []
        tempDis = -1
        for point in dataSet:
            for center in centerSet:
                tempDis += __distance(center, point)
            if tempDis > dis:
                dis = tempDis
                targetCenter = point
        centerSet.append(targetCenter)
    return centerSet

def createCluster(dataSet, k, iterTimes):
    dataMatrix = np.mat(dataSet)
    m, n = np.shape(dataMatrix)
    centerSet = __chooseInitCenter(dataSet, k)
    result = {}
    for i in range(iterTimes):
        cluster = {}
        for point in dataSet:
            dis = np.inf 
            clusterIndex = -1
            for j in range(len(centerSet)):
                center = centerSet[j]
                tempDis = __distance(point, center)
                if tempDis < dis:
                    dis = tempDis 
                    clusterIndex = j
            if clusterIndex in cluster.keys():
                cluster[clusterIndex].append(point)
            else:
                cluster[clusterIndex] = []
                cluster[clusterIndex].append(point)
        temp = 0
        for pointSet in cluster.values():
            center = __chooseClusterCenter(np.mat(pointSet))
            centerSet[temp] = center.tolist()[0]
            temp += 1
        if i == (iterTimes - 2):
            result = cluster 
    return result 
        
if __name__ == "__main__":
    dataSet = [
        [1.0,2.0],
        [3.0,6.0],
        [6.0,2.0],
        [7.0,4.0],
        [1.0,3.0],
        [2.0,3.0],
        [8.0,9.0],
        [9.0,7.0],
        [6.0,8.0],
        [5.0,6.0],
        [7.0,5.0]
    ]
xx = __chooseClusterCenter(np.mat(dataSet))
print(xx)
result = createCluster(dataSet, 2, 5)
print(result)