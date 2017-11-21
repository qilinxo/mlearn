from numpy import *
import numpy as np

# input
	# dataMatrix: ndarry, the data set.
	# inputVector: ndarray, the input vector which need to be judged for which class.
	# lables: ndarray, the lables of dataMatrix.
	# k: int, find k the most nearest items .
# output
	# targetLabel: int, the label we needed
# comment
	# find k the most nearest items, than choose the highest frequency one in these items. If frequencies are the same, return the smallest distance item.
def kNNClassify(dataMatrix, inputVector, labels, k):
	lineNoOfDataMatrix = dataMatrix.shape[0]
	multiVectors = tile(inputVector, (lineNoOfDataMatrix, 1))
	differMatrix = dataMatrix - multiVectors
	squareDifferMatrix = differMatrix ** 2
	sumSquareDifferMatrix = squareDifferMatrix.sum(axis = 1)
	distanceMatrix = sumSquareDifferMatrix ** 0.5
	print(distanceMatrix)
	indexArray = distanceMatrix.argsort()
	classCount = {}
	for i in range(k):
		labelValue = labels[indexArray[i]]
		classCount[labelValue] = classCount.get(labelValue, 0) + 1
	print(classCount)
	SortedClassCount = sorted(classCount.items(), key=lambda e:e[1], reverse=True)
	targetLabel = SortedClassCount[0][0]
	return targetLabel
	
if __name__ == "__main__":
	dataMatrix = random.random(size=(8, 5))
	print(dataMatrix)
	labels = [1,2,3,4,5,1,3,2]
	inputVector = [1,2,3,4,5]
	result = kNNClassify(dataMatrix, np.array(inputVector), np.array(labels),3)
	print(result)
		