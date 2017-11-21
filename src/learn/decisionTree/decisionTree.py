'''
Created on Nov 01, 2017

@author: Lin
'''
from math import log
import learn.decisionTree.mStringHelper as msh
import matplotlib.pyplot as plt

# input
	# dataMatrix: ndarray
	# labels: ndarray, that label vector
# output
	# entropy: float, the entropy values
# comment
	# entropy calculation: 
	# prop = number of the item / number of total items
	# entropy = -log(prop)*prop
def __calEntropy(dataSet, labels):
	noOfLines = len(dataSet)
	labelCount = {}
	entropy = 0.0
	for index in range(noOfLines):
		if labels[index] not in labelCount.keys():
			labelCount[labels[index]] = 0
		labelCount[labels[index]] += 1
	for key in labelCount.keys():
		prop = float( labelCount[key] / noOfLines )
		entropy -= log(prop)*prop
	return entropy

def __calGini(dataSet, labels):
	noOfLines = len(dataSet)
	labelCount = {}
	sumProp = 0
	for index in range(noOfLines):
		if labels[index] not in labelCount.keys():
			labelCount[labels[index]] = 0
		labelCount[labels[index]] += 1
	for key in labelCount.keys():
		prop = float(labelCount[key] / noOfLines)
		sumProp = sumProp + prop * prop
	gini = 1 - sumProp
	return gini

def __intristicValue(dataSet, labels):
	noOfLines = len(dataSet)
	labelCount = {}
	iv = 0.0
	for index in range(noOfLines):
		if labels[index] not in labelCount.keys():
			labelCount[labels[index]] = 0
		labelCount[labels[index]] += 1
	for key in labelCount.keys():
		prop = float(labelCount[key] / noOfLines)
		iv = iv - prop * log(prop)
	return iv

# input
	# dataMatrix: ndarry, the data matrix, do not contains the labels.
	# labels: ndarray, labels.
	# propertyIndex: int, perperty's index, start from 0.
# output
	# resultDataDict: dict, keys are the properties value. One key corresponds to several lines of dataMatrix.
		#{
		#	key1:[
		#		[line1],[line2],...
		#	],
		#	...
		#}
	# resultLabelDict: dict, keys are the properties value. One key corresponds to several labels.
		# {
		#	 key1:[
		#		 label1, label2,...
		#	 ],
		#	 ...
		# }
# comments
	# this method can adapt to general senario of dividing, not only the binary dividing.
def __splitDataMatrixGen(dataSet, labels, propertyIndex):
	noOfLines = len(dataSet)
	valueList = []
	resultDataDict = {}
	resultLabelDict = {}
	for index in range(noOfLines):
		if dataSet[index][propertyIndex] not in valueList:
			valueList.append(dataSet[index][propertyIndex])
			resultDataDict[dataSet[index][propertyIndex]] = []
			resultLabelDict[dataSet[index][propertyIndex]] = []
		resultDataDict[dataSet[index][propertyIndex]].append(dataSet[index])
		resultLabelDict[dataSet[index][propertyIndex]].append(labels[index])
	return resultDataDict, resultLabelDict

# input
	# resultDataDict: dict, keys are the properties value. One key corresponds to several lines of dataMatrix.
		#{
		#	key1:[
		#		[line1],[line2],...
		#	],
		#	...
		#}
	# propIndex: int, the property index
# output
	# resultDataDict: dict, the dict that exclude the target property.
def __splitDataMatrixAndExcludePropColumn(resultDataDict, propIndex):
	for key in resultDataDict:
		for i in range(len(resultDataDict[key])):
			resultDataDict[key][i].pop(propIndex)
	return resultDataDict
	
# input
	# dataSet: array, [[...],[...],...]
	# labels: array, the label data set. [...]
	# giniOn: boolean, whether use gini to split data set
	# gainRateOn: tuple, (False, -1), whether use gain rate to split data set, and the penalty factor
# output
	# resultDataDict: dict, keys are the properties value. One key corresponds to several lines of dataMatrix.
		#{
		#	key1:[
		#		[line1],[line2],...
		#	],
		#	...
		#}
	# # resultLabelDict: dict, keys are the properties value. One key corresponds to several labels.
		# {
		#	 key1:[
		#		 label1, label2,...
		#	 ],
		#	 ...
		# }
	# propIndex: int, the property index
# comment
	# choose the best property, and split the data set.
def __chooseBestProAndSplit(dataSet, labels, giniOn=False, gainRateOn=False):
	noOfLines = len(dataSet)
	noOfProp = len(dataSet[0])
	totalEntropy = 0
	if giniOn == False:
		totalEntropy = __calEntropy(dataSet, labels)
	else:
		totalEntropy = __calGini(dataSet, labels)
	gainEntropyOrGini = 0.0
	propIndex = 0
	resultDataDict = {}
	resultLabelDict = {}
	if giniOn == False:
		for index in range(noOfProp):
			temptDataDict, temptLabelDict = __splitDataMatrixGen(dataSet, labels, index)
			temptGainEntropy = 0.0
			temptConditionalEntropy = 0.0
			for key in temptDataDict:
				temptConditionalEntropy += ( __calEntropy(temptDataDict[key], temptLabelDict[key]) * len(temptLabelDict[key]) / noOfLines )
			if gainRateOn == False:
				temptGainEntropy = totalEntropy - temptConditionalEntropy
			else:
				temptGainEntropy = (totalEntropy - temptConditionalEntropy) * __intristicValue(temptDataDict[key], temptLabelDict[key])
			if temptGainEntropy >= gainEntropyOrGini:
				gainEntropyOrGini = temptGainEntropy
				propIndex = index
				resultDataDict = temptDataDict
				resultLabelDict = temptLabelDict
		return resultDataDict, resultLabelDict, propIndex
	else:
		for index in range(noOfProp):
			temptDataDict, temptLabelDict = __splitDataMatrixGen(dataSet, labels, index)
			temptGini = 0.0
			for key in temptDataDict:
				temptGini += ( __calGini(temptDataDict[key], temptLabelDict[key]) * len(temptLabelDict[key]) / noOfLines )
			if temptGini <= gainEntropyOrGini or gainEntropyOrGini - 0 < 0.0000001:
				gainEntropyOrGini = temptGini
				propIndex = index
				resultDataDict = temptDataDict
				resultLabelDict = temptLabelDict
			return resultDataDict, resultLabelDict, propIndex
		
# input		
	# labels: array, the labels input
# output
	# the label who has largest number
def __maxNumOfLabel(labels):
	labelDict = {}
	for label in labels:
		if label not in labelDict.keys():
			labelDict[label] = 0
		labelDict[label] += 1
	resultArray = sorted(labelDict.items(), key=lambda d:d[1], reverse = True)
	return resultArray[0][0]
	
# input
	# dataSet: array, the data set. [[],[],...]
	# labels: array, the data set's labels. []
	# properties: array, the properties array. []
	# giniOn: boolean, whether use gini to split data set
	# gainRateOn: tuple, (False, -1), whether use gain rate to split data set, and the penalty factor
# output
	# myTree: the tree we need. for excample: 
		# {'Outlook': {'sunny': {'Humidity': {'high': {'Windy': {'false': 'no', 'true': 'no'}}, 'normal': 'yes'}}, 'overcast': 'yes', 'rain': 'yes'}}
def createTree(dataSet, labels, properties, giniOn=False, gainRateOn=False, maxDepth=-1):
	resultDataDict, resultLabelDict, propIndex = __chooseBestProAndSplit(dataSet, labels, giniOn, gainRateOn)
	choosenProp = properties[propIndex]
	myTree = {choosenProp:{}}
	resultDataDict = __splitDataMatrixAndExcludePropColumn(resultDataDict, propIndex)
	properties.pop(propIndex)
	for key in resultDataDict.keys():
		if maxDepth < 0:
			if len(properties) > 1:
				myTree[choosenProp][key] = createTree(resultDataDict[key], resultLabelDict[key], properties, giniOn, gainRateOn, maxDepth-1)
			elif len(properties) == 1:
				myTree[choosenProp][key] = __maxNumOfLabel(resultLabelDict[key])
		else:
			if len(properties) > maxDepth:
				myTree[choosenProp][key] = createTree(resultDataDict[key], resultLabelDict[key], properties, giniOn, gainRateOn, maxDepth-1)
			elif len(properties) == maxDepth:
				myTree[choosenProp][key] = __maxNumOfLabel(resultLabelDict[key])
	return myTree

# input
# 	dataVector: array, [], the data line should be judge for which class is.
# 	properties: array, [], the propeties array corresponding to the data vector.
# 	myTree: dict, the decision tree we trained.
# output
# 	label: str, the class label that the input data line belongs to
def dataClassify(dataVector, properties, myTree):
	label = ''
	firstProp = list(myTree.keys())[0]
	propIndex = properties.index(firstProp)
	for key in myTree[firstProp].keys():
		if type(myTree[firstProp][key]).__name__ == 'dict' and dataVector[propIndex] == key:
			label = dataClassify(dataVector, properties, myTree[firstProp][key])
			break
		elif type(myTree[firstProp][key]).__name__ != 'dict' and dataVector[propIndex] == key:
			label = myTree[firstProp][key]
			break
		else:
			continue
	return label

# input
# 	myTree: dict, the decision tree, for excample:
# 	{'Outlook': {'sunny': {'Humidity': {'high': {'Windy': {'false': 'no', 'true': 'no'}}, 'normal': 'yes'}}, 'overcast': 'yes', 'rain': 'yes'}}
# output
# 	noOfleafs: int, the no of leafs
def __getNoOfLeafs(myTree):
	noOfleafs = 0
	for key in myTree.keys():
		if type(myTree[key]).__name__ == 'dict':
			noOfleafs += __getNoOfLeafs(myTree[key])
		else:
			noOfleafs += 1
	return noOfleafs
	
# input
# 	myTree: dict, the decision tree, for excample:
# 	{'Outlook': {'sunny': {'Humidity': {'high': {'Windy': {'false': 'no', 'true': 'no'}}, 'normal': 'yes'}}, 'overcast': 'yes', 'rain': 'yes'}}
# output
# 	noOfleafs: int, the depth of decision tree
def __getDepthOfTree(myTree):
	depthOfTree = 1
	for key in myTree.keys():
		tempDepth = 1
		for key2 in myTree[key].keys():
			if type(myTree[key][key2]).__name__ == 'dict':
				tempDepth += __getDepthOfTree(myTree[key][key2])					
				if tempDepth > depthOfTree:
					depthOfTree = tempDepth
				else:
					depthOfTree += 1
	return depthOfTree

# input
# 	childPosition: tuple, (x, y), the child node position
# 	fatherPosition: tuple, (x, y), the father node positon
# 	txt: String, the text on the side
# comment
# 	use matplotlib to draw the side and the text on the side
def __drawMidTxtAndSide(childPosition, fatherPosition, midtextPosition, txt):
	plt.subplot(111, frameon=False).text(midtextPosition[0], midtextPosition[1], txt,va='center', ha='center')
	plt.subplot(111, frameon=False).annotate("", xytext=childPosition, xy=fatherPosition, va='center', ha='center', arrowprops=dict(arrowstyle='<-'))

# input
# 	position: tuple, (x, y), the position of the node should draw
#	txt: str, the text on the node
# comment
# 	use matplotlib to draw the node and the text on this node
def __drawNode(position, txt):
	plt.subplot(111, frameon=False).annotate(txt, xy=position, bbox=dict(boxstyle='round4', fc='0.8'), va='center', ha='center')
	
# input
# 	myTree: dict, the target decision tree. for excample:
#		{'Outlook': {'sunny': {'Humidity': {'high': {'Windy': {'false': 'no', 'true': 'no'}}, 'normal': 'yes'}}, 'overcast': 'yes', 'rain': 'yes'}}
# 	initNodePosition: tuple. (x, y). the first node of the decision tree.
# 	unitX: int, the x axis unit length, it should be the wide of subgraph divide the No. of leafs
# 	unitY: int, the y axis unit length, it should be the depth of subgraph divede the no. of tree depth
# comment
# 	use matplotlib to draw the whole decision tree
def __drawTree(myTree, initNodePosition, unitX, unitY):
	firstTxt = list(myTree.keys())[0]
	fatherPosition = initNodePosition
	__drawNode(fatherPosition, firstTxt)
	keySize = len(list(myTree[firstTxt].keys()))
	childPositionX = fatherPosition[0] - (keySize/2 * unitX)
	midSum = 0
	for key in myTree[firstTxt].keys():
		secondTxt = myTree[firstTxt][key]
		midTxt = key
		childPosition = (childPositionX, fatherPosition[1] - unitY)
		midtextPosition = ((childPosition[0] + fatherPosition[0]) / 2, (childPosition[1] + fatherPosition[1]) / 2 - (midSum/8*unitY))
		__drawMidTxtAndSide(childPosition, fatherPosition, midtextPosition, midTxt)
		midSum += 1
		childPositionX = childPositionX + unitX
		if type(myTree[firstTxt][key]).__name__ == 'dict':
			childTree = myTree[firstTxt][key]
			__drawTree(childTree, childPosition, unitX, unitY)
		else:
			__drawNode(childPosition, secondTxt)

# input
# 	myTree: dict, the target decision tree. for excample:
#		{'Outlook': {'sunny': {'Humidity': {'high': {'Windy': {'false': 'no', 'true': 'no'}}, 'normal': 'yes'}}, 'overcast': 'yes', 'rain': 'yes'}}
# commment
#	show the graph
def showTree(myTree):
	wide = __getNoOfLeafs(myTree)
	depth = __getDepthOfTree(myTree)
	unitX = 1/2/wide
	unitY = 1/depth
	__drawTree(myTree, (0.5, 1.0), unitX, unitY)
	plt.show()
	
	
if __name__ == "__main__":
	outPutMatrix, lableVector, properties = msh.importDataSet("c:\\M_learn\\dicisionTreeDataSet")
	print(outPutMatrix, lableVector, properties)
	#print(__calEntropy(outPutMatrix, lableVector))
	#print(__calGini(outPutMatrix, lableVector))
# 	#resultDataDict, resultLabelDict, targetPro = __chooseBestProAndSplit([['hot', 'false'], ['hot', 'true'], ['mild', 'false']], ['no', 'no', 'no'])
# 	#print(resultDataDict, resultLabelDict, targetPro)
	myTree = createTree(outPutMatrix, lableVector, properties, False, True)
	#print(myTree)
	label = dataClassify(['hot', 'sunny', 'high', 'true'], ['Temperatrue', 'Outlook', 'Humidity', 'Windy'], myTree)
	print(label)
	#myTree = {'Outlook': {'sunny': {'Humidity': {'high': {'Windy': {'false': 'no', 'true': 'no'}}, 'normal': 'yes'}}, 'overcast': 'yes', 'rain': 'yes'}}
#	print(myTree)
# 	print(__getNoOfLeafs(myTree))
# 	print(__getDepthOfTree(myTree))
# 	#drawNode((0.5,0.5), "kaka")
# 	#plt.show()
# 	drawMidTxtAndSide((0.5,0.5),(0,0), "kaka")
# 	plt.show()
	showTree(myTree)