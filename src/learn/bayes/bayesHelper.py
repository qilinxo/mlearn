'''
Created on Nov 13, 2017

@author: Lin
'''
def createDataSet():
    wordsList = [['I','am','a','teacher','we','need','meachine','learning'],\
                 ['what','a','fucking','are','you','what','do','you','want','to','do'],\
                 ['how','stupit','you','are','i','am','your','father'],\
                 ['meachine','learning','is','wonderful'],\
                 ['i','love','choose'],\
                 ['haw','fuck','you','are','what','are','you','talking','about']]
    labelVector = [0, 1, 1, 0, 0, 1]
    return wordsList, labelVector

def createVocabList(dataSet):
    vocabSet = set([])
    for statement in dataSet:
        for word in statement:
            vocabSet.add(word)
    return list(vocabSet)

def wordsToVectoers(vocabList, wordsList):
    returnMatrix = []
    for vec in wordsList:
        returnVec = [0] * len(vocabList)
        for word in vec:
            index = 0
            for item in vocabList:
                if item == word:
                    returnVec[index] = 1
                index += 1
        returnMatrix.append(returnVec)
    return returnMatrix

if __name__ == '__main__':
    wordsList, labelVector = createDataSet()
    vcList = createVocabList(wordsList)
    print(wordsList)
    print(vcList)
    print(wordsToVectoers(vcList, wordsList))