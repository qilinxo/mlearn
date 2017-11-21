import numpy as np
import matplotlib.pyplot as plt
import datetime
# input： 
	# fileName: String, the element in this array should be int
	# separator: String, can be "\t"..., default is " "
	# labelColumn: int, label is in which column in dataset file. default is the last column. value range from 0.
# output:
	# outPutMatrix: an ndarray matrix
	# lableVector: array
def importFileToMatrix(fileName, separator=" ", labelColumn=-1):
	fr = open(fileName)
	arrayOfLines = fr.readlines()
	numberOfLines = len(arrayOfLines)
	lableVector = []
	if(numberOfLines > 0):
		numberOfColumns = len(arrayOfLines[0].split(separator))
	outPutMatrix = np.zeros((numberOfLines, numberOfColumns-1))
	index = 0
	for line in arrayOfLines:
		lineItems = line.split(separator)
		lineItems[-1] = lineItems[-1].rstrip("\n")
		if labelColumn == -1:
			outPutMatrix[index, :] = lineItems[0:numberOfColumns-1]
			lableVector.append(float(lineItems[numberOfColumns-1]))
		elif labelColumn == 0:
			outPutMatrix[index, 1:numberOfColumns] = lineItems[1:numberOfColumns]
			lableVector.append(float(lineItems[0]))
		else:
			outPutMatrix[index, 0:labelColumn] = lineItems[0:labelColumn]
			outPutMatrix[index, labelColumn+1:numberOfColumns] = lineItems[labelColumn+1:numberOfColumns-1]
			lableVector.append(float(lineItems[labelColumn]))
		index += 1
	return outPutMatrix, lableVector
	
# input:
	# dataMatrix: ndarray
	# xColumn: int, default is 1
	# yColumn: int, default is 2
# comment:
	# only can figure two properties in one matrix
def dataDrawer(dataMatrix, xColumn=1, yColumn=2):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(dataMatrix[:, xColumn], dataMatrix[:, yColumn])
	plt.show()

# input
# 	dataVector: [[],[],...]
# 	labelVector: []
# comment
# 	when the label and data are split into two data set, use this function
def dataAndLabelDrawer(dataVector, labelVector):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	dataVector = np.array(dataVector).transpose()
	ax.scatter(dataVector, labelVector)
	plt.show()
	
# input:
	# dataMatrix: ndarray
# comment: nomalize
	# (value - min) / (max - min)
def nomalize(dataMatrix):
	numberOfLines = len(dataMatrix)
	minVals = dataMatrix.min(0)
	maxVals = dataMatrix.max(0)
	rangeVals = maxVals - minVals
	rangeMatrix = dataMatrix - np.tile(minVals, (numberOfLines, 1))
	nomalizeMatrix = rangeMatrix / np.tile(rangeVals, (numberOfLines, 1))
	return nomalizeMatrix

# input
	# dataMatrix: ndarray, the target matrix set
	# percent: int, the percent that the testing set take percents of the whole data set. For Example, 10 is stands for 10%.
# output
	# trainMatrix: ndarray
	# testMatrix: ndarray
# comment
	# for example, the input percent is 10. the method will divide the first 90% data into training set, and the last 10% data into test set.
def divideTrainSetFirst(dataMatrix, percent):
	numberOfLines = dataMatrix.shape[0]
	numberOfTestLines = int(numberOfLines * percent / 100)
	numberOfTrainLines = numberOfLines - numberOfTestLines
	trainMatrix = dataMatrix[: numberOfTrainLines, :]
	testMatrix = dataMatrix[numberOfTrainLines :, :]
	return trainMatrix, testMatrix

# input
# 	func: the function we need test
# 	*args: the values the function need input
# output:
# 	exeTime: the execution time for the function
# 	result: if the function output multiple values, the type of result is set (). if the function output single value, the type of result is the ouput type.
# comment:
# 	output the delta time the target function spend.
def timeSpend(func, *args):
	starttime = datetime.datetime.now()
	result = func(*args)
	endtime = datetime.datetime.now()
	exeTime = (endtime - starttime).microseconds 
	return exeTime, result

# input
	# dataMatrix: ndarray, the target matrix set
	# percent: int, the percent that the testing set take percents of the whole data set. For Example, 10 is stands for 10%.
# output
	# trainMatrix: ndarray
	# testMatrix: ndarray
# comment
	# for example, the input percent is 10. the method will divide the first 10% data into testing set, and the last 90% data into training set.
def divideTestSetFirst(dataMatrix, percent):
	numberOfLines = dataMatrix.shape[0]
	numberOfTestLines = int(numberOfLines * percent / 100)
	testMatrix = dataMatrix[: numberOfTestLines, :]
	trainMatrix = dataMatrix[numberOfTestLines :, :]
	return trainMatrix, testMatrix
	
if __name__ == '__main__':
	outPutMatrix, lableVector = importFileToMatrix("c:\\M_learn\\dataset.txt")
	print(outPutMatrix, lableVector)
	#dataDrawer(outPutMatrix)
	print(nomalize(outPutMatrix))
	trainMatrix, testMatrix = divideTestSetFirst(outPutMatrix, 50)
	print(trainMatrix, testMatrix)
	
	
	
	
