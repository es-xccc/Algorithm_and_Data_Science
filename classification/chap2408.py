# -*- coding: utf-8 -*-
"""
Created on Sat May 14 18:04:35 2022

@author: tiw
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:08:15 2020

@author: tiw
"""
import random
import pylab
import math

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

def cosine_similarity(v1,v2):
    #compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

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

def genderofRunners(examples):
    Male,Female=0,0
    for r in examples:
        if r.getLabel()=='M':
            Male+=1
        else:
            Female+=1
    return Male, Female

#Figure 24.6
def findKNearest(example, exampleSet, k):
    kNearest, distances = [], []
    #Build lists containing first k examples and their distances
    for i in range(k):
        kNearest.append(exampleSet[i])
        distances.append(example.featureDist(exampleSet[i]))
    maxDist = max(distances) #Get maximum distance
    #Look at examples not yet considered
    for e in exampleSet[k:]:
        dist = example.featureDist(e)
        if dist < maxDist:
            #replace farther neighbor by this one
            maxIndex = distances.index(maxDist)
            kNearest[maxIndex] = e
            distances[maxIndex] = dist
            maxDist = max(distances)      
    return kNearest, distances

def findKNearestCS(example, exampleSet, k):
    kNearest, similarities = [], []
    #Build lists containing first k examples and their distances
    for i in range(k):
        kNearest.append(exampleSet[i])
        similarities.append(example.cosine_similarity(exampleSet[i]))
    maxSim = max(similarities) #Get maximum distance
    #Look at examples not yet considered
    for e in exampleSet[k:]:
        sim = example.cosine_similarity(e)
        if sim < maxSim:
            #replace farther neighbor by this one
            maxIndex = similarities.index(maxSim)
            kNearest[maxIndex] = e
            similarities[maxIndex] = sim
            maxSim = max(similarities)      
    return kNearest, similarities

def KNearestClassify(training, testSet, label, k):
    """Assumes training and testSet lists of examples, k an int
       Uses a k-nearest neighbor classifier to predict
         whether each example in testSet has the given label
       Returns number of true positives, false positives,
          true negatives, and false negatives"""
    truePos, falsePos, trueNeg, falseNeg = 0, 0, 0, 0
    for e in testSet:
#        nearest, distances = findKNearest(e, training, k)
        nearest, similarities = findKNearest(e, training, k)
        #conduct vote
        numMatch = 0
        for i in range(len(nearest)):
            if nearest[i].getLabel() == label:
                numMatch += 1
        if numMatch > k//2: #guess label
            if e.getLabel() == label:
                truePos += 1
            else:
                falsePos += 1
        else: #guess not label
            if e.getLabel() != label:
                trueNeg += 1
            else:
                falseNeg += 1
    return truePos, falsePos, trueNeg, falseNeg


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

def confusionMatrix(truePos, falsePos, trueNeg, falseNeg, k):
    print('\nk = ', k)
    print('TP,FP,TN,FN = ', truePos, falsePos, trueNeg, falseNeg)
    print('                     ', 'TP', '\t\t\t', 'FP','\t\t\t', 'TP', ' \t', 'FP' )
    print('Confusion Matrix is: ', truePos,'/',MT, '\t', falsePos, '/', MT,'\t', round(truePos/MT,2), '\t', round(falsePos/MT,2))
    print('                     ', trueNeg, '/', FT, '\t', falseNeg, '/', FT,'\t', round(trueNeg/FT,2), '\t', round(falseNeg/FT,2))
    print('                     ', 'TN', '\t\t\t', 'FN','\t\t\t', 'TN', ' \t', 'FN' )    
    getStats(truePos, falsePos, trueNeg, falseNeg)
    return

def findK(training, minK, maxK, numFolds, label): 
    #Find average accuracy for range of odd values of k 
    accuracies = [] 
    for k in range(minK, maxK + 1, 2): 
        score = 0.0 
        for i in range(numFolds): #downsample to reduce computation time
            fold = random.sample(training, min(5000, len(training))) 
            examples, testSet = divide80_20(fold) 
            truePos, falsePos, trueNeg, falseNeg = KNearestClassify(examples, testSet, label, k) 
            score += accuracy(truePos, falsePos, trueNeg, falseNeg) 
#            confusionMatrix(truePos, falsePos, trueNeg, falseNeg, k)
        accuracies.append(score/numFolds) 
    return accuracies

##Code at bottom of page 410
##Note that this takes a bit of time to run
examples = buildMarathonExamples('bm_results2012.txt')
trainingSet, testSet=divide80_20(examples)
M,F = genderofRunners(trainingSet)
MT,FT=genderofRunners(testSet)
print('Training Set (M,F): ', (M,F),'\n','Testing Set (M,F): ', (MT,FT))


minK, maxK, numFolds, label = 1, 25, 5, 'F'
accuracies = findK(trainingSet, minK, maxK, numFolds, label) 

realaccuracies = []
for k in range(minK, maxK + 1, 2): 
    truePos, falsePos, trueNeg, falseNeg = KNearestClassify(trainingSet, testSet, label, k)
    realaccuracies.append(accuracy(truePos, falsePos, trueNeg, falseNeg))
    
pylab.xticks(range(minK, maxK+1, 2))
pylab.plot(range(minK, maxK + 1, 2), accuracies) 
pylab.plot(range(minK, maxK + 1, 2), realaccuracies) 
pylab.title('Average Accuracy vs k (' + str(numFolds) + ' folds)')
pylab.xlabel('k') 
pylab.ylabel('Accuracy') 
pylab.show()


#truePos, falsePos, trueNeg, falseNeg = KNearestClassify(trainingSet, testSet, 'M', 9)
#confusionMatrix(truePos, falsePos, trueNeg, falseNeg)

##Code at bottom of page 412
#reducedtrainingSet = random.sample(trainingSet, len(trainingSet)//10) #Reduce for quick testing   
#truePos, falsePos, trueNeg, falseNeg = KNearestClassify(reducedtrainingSet, testSet, 'M', 9)
#confusionMatrix(truePos, falsePos, trueNeg, falseNeg)