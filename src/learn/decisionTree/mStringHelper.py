# input
	# fileName: String, the path of input data file
	# separator: String,  can be "\t"..., default is " "
	# labelColumn: int, label is in which column in dataset file. default is the last column. value range from 0.
# output:
	# dataset: [[],[],[],...]
	# labelSet: [...]
	# properties: [...]
def importDataSet(fileName, separator=" ", labelColumn=-1):
	file = open(fileName)
	arrayOfLines = file.readlines()
	dataSet = []
	labelSet = []
	properties = arrayOfLines[0].rstrip("\n").split(separator)
	properties.pop(labelColumn)
	for line in arrayOfLines[1:]:
		line = line.rstrip("\n")
		lineItems = line.split(separator)
		labelSet.append(lineItems[labelColumn])
		lineItems.pop(labelColumn)
		dataSet.append(lineItems)
	return dataSet, labelSet, properties

