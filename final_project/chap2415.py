# -*- coding: utf-8 -*-
"""
Created on Mon May 30 17:19:50 2022

@author: tiw
"""
import random
import pylab
import math
import sklearn.linear_model #https://scikit-learn.org/stable/
               #https://en.wikipedia.org/wiki/Logistic_regression
               
def variance(X):
    """Assumes that X is a list of numbers.
       Returns the standard deviation of X"""
    mean = sum(X)/len(X)
    tot = 0.0
    for x in X:
        tot += (x - mean)**2
    return tot/len(X)
    
def stdDev(X):
    """Assumes that X is a list of numbers.
       Returns the standard deviation of X"""
    return variance(X)**0.5

def getBMData(filename):
    """Read the contents of the given file. Assumes the file 
    in a comma-separated format, with 6 elements in each entry:
    0. Name (string), 1. Gender (string), 2. Age (int)
    3. Division (int), 4. Country (string), 5. Overall time (float)   
    Returns: dict containing a list for each of the 6 variables."""
    data = {}
    f = open(filename)
    line = f.readline() 
    data['name'], data['gender'], data['age'] = [], [], []
    data['division'], data['country'], data['time'] = [], [], []
    while line != '':
        split = line.split(',')
        data['name'].append(split[0])
        data['gender'].append(split[1])
        data['age'].append(int(split[2]))
        data['division'].append(int(split[3]))
        data['country'].append(split[4]) 
        data['time'].append(float(split[5][:-1])) #remove \n
        line = f.readline()
    f.close()
    maleTime, femaleTime = [], []
    for i in range(len(data['time'])):
        if data['gender'][i]=='M':
            maleTime.append(data['time'][i])
        else:
            femaleTime.append(data['time'][i])
    print(len(maleTime),' Males and', len(femaleTime),'Females')    
    return data, maleTime, femaleTime

class Runner(object): 
    def __init__ (self, gender, age, time): 
        self.featureVec = (age, time) 
        self.label = gender 

    def featureDist(self, other): 
        dist = 0.0 
        for i in range(len(self.featureVec)): 
            dist += abs(self.featureVec[i] - other.featureVec[i])**2 
        return dist**0.5 
    
    def cosine_similarity(self,other):
    #compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(self.featureVec)):
            x = self.featureVec[i]; y = other.featureVec[i]
            sumxx += x*x
            sumyy += y*y
            sumxy += x*y
        return sumxy/math.sqrt(sumxx*sumyy)
    
    def getTime(self): 
        return self.featureVec[1] 

    def getAge(self): 
        return self.featureVec[0] 

    def getLabel(self): 
        return self.label 

    def getFeatures(self): 
        return self.featureVec 

    def __str__ (self): 
        return str(self.getAge()) + ', ' + str(self.getTime()) + ', ' + self.label
    
def buildMarathonExamples(fileName): 
#   data, maleTime, femalTime = getBMData(fileName)
   data, maleTime, FemaleTime = getBMData(fileName)
   examples = [] 
   for i in range(len(data['age'])): 
       a = Runner(data['gender'][i], data['age'][i], 
                  data['time'][i]) 
       examples.append(a)
   return examples 

def makeHist(data, bins, title, xLabel, yLabel):
    pylab.hist(data, bins, edgecolor='black')
    pylab.title(title)
    pylab.xlabel(xLabel)
    pylab.ylabel(yLabel)
    mean = sum(data)/len(data)
    std = stdDev(data)
    pylab.annotate('Mean = ' + str(round(mean, 2)) +\
              '\nSD = ' + str(round(std, 2)), fontsize = 20,
              xy = (0.50, 0.75), xycoords = 'axes fraction')                         

def divide80_20(examples): 
    sampleIndices = random.sample(range(len(examples)), len(examples)//5) 
    trainingSet, testSet = [], [] 
    for i in range(len(examples)): 
        if i in sampleIndices: 
            testSet.append(examples[i]) 
        else: trainingSet.append(examples[i]) 
    return trainingSet, testSet 


def accuracy(truePos, falsePos, trueNeg, falseNeg): 
    numerator = truePos + trueNeg 
    denominator = truePos + trueNeg + falsePos + falseNeg 
    return numerator/denominator 
def sensitivity(truePos, falseNeg): 
    try: 
        return truePos/(truePos + falseNeg) 
    except ZeroDivisionError: 
        return float('nan') 
def specificity(trueNeg, falsePos): 
    try: 
        return trueNeg/(trueNeg + falsePos) 
    except ZeroDivisionError: 
        return float('nan') 
def posPredVal(truePos, falsePos): 
    try:
        return truePos/(truePos + falsePos) 
    except ZeroDivisionError: 
        return float('nan') 
def negPredVal(trueNeg, falseNeg): 
    try: 
        return trueNeg/(trueNeg + falseNeg) 
    except ZeroDivisionError: 
        return float('nan') 

def getStats(truePos, falsePos, trueNeg, falseNeg, toPrint = True): 
    accur = accuracy(truePos, falsePos, trueNeg, falseNeg) 
    sens = sensitivity(truePos, falseNeg) 
    spec = specificity(trueNeg, falsePos) 
    ppv = posPredVal(truePos, falsePos) 
    if toPrint: 
        print(' Accuracy =', round(accur, 3)) 
        print(' Sensitivity =', round(sens, 3)) 
        print(' Specificity =', round(spec, 3)) 
        print(' Pos. Pred. Val. =', round(ppv, 3)) 
    return (accur, sens, spec, ppv) 

def genderofRunners(examples):
    Male,Female=0,0
    for r in examples:
        if r.getLabel()=='M':
            Male+=1
        else:
            Female+=1
    return Male, Female

def confusionMatrix(truePos, falsePos, trueNeg, falseNeg):
#    print('\nk = ', k)
    print('TP,FP,TN,FN = ', truePos, falsePos, trueNeg, falseNeg)
    print('                     ', 'TP', '\t\t\t', 'FP','\t\t\t', 'TP', ' \t', 'FP' )
    print('Confusion Matrix is: ', truePos,'/',MT, '\t', falsePos, '/', MT,'\t', round(truePos/MT,2), '\t', round(falsePos/MT,2))
    print('                     ', trueNeg, '/', FT, '\t', falseNeg, '/', FT,'\t', round(trueNeg/FT,2), '\t', round(falseNeg/FT,2))
    print('                     ', 'TN', '\t\t\t', 'FN','\t\t\t', 'TN', ' \t', 'FN' )    
    getStats(truePos, falsePos, trueNeg, falseNeg)
    return
    
def applyModel(model, testSet, label, prob = 0.5):
    #Create vector containing feature vectors for all test examples
    testFeatureVecs = [e.getFeatures() for e in testSet]
    probs = model.predict_proba(testFeatureVecs)
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for i in range(len(probs)):
        if probs[i][1] > prob:
            if testSet[i].getLabel() == label:
                truePos += 1
            else:
                falsePos += 1
        else:
            if testSet[i].getLabel() != label:
                trueNeg += 1
            else:
                falseNeg += 1
    return truePos, falsePos, trueNeg, falseNeg

examples = buildMarathonExamples('bm_results2012.txt')
trainingSet, testSet = divide80_20(examples)
M,F = genderofRunners(trainingSet)
MT,FT=genderofRunners(testSet)
#print('Training Set (M,F): ', (M,F),'\n','Testing Set (M,F): ', (MT,FT))
#
featureVecs, labels = [], []
for e in trainingSet:
    featureVecs.append([e.getAge(), e.getTime()])
    labels.append(e.getLabel()) #'M' or 'F'
model = sklearn.linear_model.LogisticRegression().fit(featureVecs, labels)
print('Feature weights for label M:',
      'age =', str(round(model.coef_[0][0], 3)) + ',',
      'time =', round(model.coef_[0][1], 3))
truePos, falsePos, trueNeg, falseNeg = applyModel(model, testSet, 'M', 0.5)
confusionMatrix(truePos, falsePos, trueNeg, falseNeg)


allAccuracy, alltruePos, allfalsePos, alltrueNeg, allfalseNeg= [],[],[],[],[]
count=0
maxAccuracy=0
maxcount=0
maxk=0
for k in range(500, 601, 1):
    truePos, falsePos, trueNeg, falseNeg = applyModel(model, testSet, 'M', k/1000)
#    print('k=',k/1000, 'yields: ',truePos, falsePos, trueNeg, falseNeg, '\n') 
    alltruePos.append(truePos)
    allfalsePos.append(falsePos)
    alltrueNeg.append(trueNeg)
    allfalseNeg.append(falseNeg)
    accur=accuracy(truePos, falsePos, trueNeg, falseNeg) 
    allAccuracy.append(accur)
    if maxAccuracy < accur:
        maxAccuracy = accur
        maxcount=count
        maxk=k/1000
    count+=1
print('All TruePos: \n', alltruePos)    
avetruePos=sum(alltruePos)/count
avefalsePos=sum(allfalsePos)/count
avetrueNeg=sum(alltrueNeg)/count 
avefalseNeg=sum(allfalseNeg)/count    
kValues=[k/1000 for k in range(500, 601, 1)]
confusionMatrix(int(avetruePos), int(avefalsePos), int(avetrueNeg), int(avefalseNeg))
truePos, falsePos, trueNeg, falseNeg = applyModel(model, testSet, 'M', maxk)
print('\n\nConfusion matrix for maxmum k ', maxk, 'at ', maxcount, '\n' )
confusionMatrix(truePos, falsePos, trueNeg, falseNeg)
#Below line is checking only
#print('k at: ', maxk, 'with accuracy: ', accuracy(alltruePos[maxcount], allfalsePos[maxcount], alltrueNeg[maxcount], allfalseNeg[maxcount]))
pylab.plot(kValues, allAccuracy)
pylab.plot(maxk, allAccuracy[maxcount],'ro')
pylab.annotate((maxk, round(allAccuracy[maxcount],3)), xy=(maxk,allAccuracy[maxcount]))
pylab.title('Threshold values vs Accuracies')
pylab.xlabel('Threshold Values k')
pylab.ylabel('Accuracy')
pylab.show()