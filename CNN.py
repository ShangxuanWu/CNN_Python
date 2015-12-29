#-*-coding:utf-8-*-

#CopyRight Shangxuan Wu @ SYSU
#2015/12/26

from numpy import *
import sys,os
import math
import pdb
#parameters, modify it to meet your need
trainEpoch = 1000
w1 = random.rand(4,4)
b1 =  random.rand(4,1)
w2 = random.rand(4,4)
b2 = random.rand(4,1)
theta = random.rand(4,3)
b3 = zeros([3,1])#the bias of last layer is always zero
baseLR = 0.01
momentum = 0.9
labelNum = 3
#network shape:
#x-> fc1->fc2->softmax

#w1#w2 #\theta#-| 
#-> #-> #->     #  |->softmax
#b1# b2 #         # -|
#    #     #

def readAllFile(x):
    totalNum = 0
    totalSet = []
    totalLabel = []
    path = sys.path[0]   
    f = open(path+'\\dataset\\'+x, "r")  
    while True:  
        line = f.readline()  
        if line:  
            #print line
            a = line.split(",")
                
            if a[4] == 'Iris-versicolor\n':
                a[4] = 1
            elif a[4] == 'Iris-virginica\n':
                a[4] = 2
            else:
                a[4] = 0
            a[0]=float(a[0])
            a[1]=float(a[1])
            a[2]=float(a[2])
            a[3]=float(a[3])
            print a
            totalLabel.append(a[4])
            del a[4]
            totalSet.append(a)
            totalNum = totalNum + 1
            print totalNum
            pass    # do something here   
        else:  
            break
    f.close()
    return totalSet,totalNum,totalLabel

def splitTrainTest(totalNum):#还要加一个乱序的功能
    anchor = random.rand(totalNum,1)
#    print anchor.dtype.name
    for var in nditer(anchor, op_flags=['readwrite']):
        if var >= 0.5:
            var[...]= 1
        else:
            var[...] = 0    
    return anchor

def computeLoss(output, label):#Softmax-Loss
    allLabel = zeros((labelNum, 1))
    allLabel[label]=1
    
    loss = 0.0
    total = 0.0
    softmax = zeros(shape(output))
    # pdb.set_trace()
    # for index in range(labelNum):
        # loss = loss + (output[index]-allLabel[index])**2
    for index in range(labelNum):
        softmax[index] = math.exp(output[index])
        total = total + softmax[index]
    
    softmax = softmax / total
    for index in range(labelNum):
        loss = loss - math.log(softmax[index])
    print 'Probability: '
    print softmax
    print 'loss: '
    print loss
    # math.log()
    # print 'loss = '
    # print loss
    return softmax, loss
    
def forward_and_backward(x, label):
    
    global theta
    global b3
    global w2
    global b2
    global w1
    global b1
    
    x=asarray(x)  
    x.shape=(4,1)    
    blob1 = dot(transpose(w1),x)+b1
    blob2 = dot(transpose(w2),blob1)+b1
    output = dot(transpose(theta),blob2)+b3
    tempLabel = argmax(output)
    
    [softmax, loss] = computeLoss(output, label)
    return tempLabel, loss
    
    #compute the gradient of the last fc and softmax layer
    #x(i)is 4-d 
    DeltaTheta = zeros([4,3])
    for index1 in range(4):
        for index2 in range(3):
        #blob2 should be 4*1
            if index2 == label:
                one = 1
            else:
                one = 0
            difference = one - softmax[index2]
            temp = blob2*difference
            DeltaTheta[:,index] = temp
    #update the last softmax layer
    theta = theta - baseLR*(DeltaTheta+momentum*theta)
    
    delta3 = -(y - a3)
    #compute the gradient of the second fc layer
    
    deltaW2 = dot(delta3, transpose(a2))
    deltaB2 =  delta3
    
    #update the second fc layer 
    
    delta2 = dot(w2, delta3)
    
    w2 = w2 - baseLR*(DeltaW2+momentum*w2)
    b2 = b2 - baseLR*DeltaB2
    
    #compute the gradient of the first fc layer
    
    #update the first fc layer
    DeltaW1 = dot(delta2,transpose(a1))
    DeltaB1 = delta2
    w1 = w1 - baseLR*(DeltaW1+momentum*w1)
    b1 = b1 - baseLR*DeltaB1

def forward_test(x, realLabel):
    x=asarray(x)  
    x.shape=(4,1)    
    blob1 = dot(transpose(w1),x)+b1
    blob2 = dot(transpose(w2),blob1)+b1
    output = dot(transpose(theta),blob2)+b3
    label = argmax(output)
    [softmax, loss] = computeLoss(output, realLabel)
    return label, softmax

def train(trainSet, trainLabel):
    trainNum = len(trainSet)
    
    for epoch in range(trainEpoch):
        print 'epoch: '
        print trainEpoch
        for dataNum in range(trainNum):
            # pdb.set_trace()
            [predictLabel, loss] = forward_and_backward(trainSet[dataNum], trainLabel[dataNum])
            print 'Pic: '
            print dataNum
            print 'loss: '
            print loss
            # backward(loss)
    print 'this epoch train complete'

def test(testSet, testLabel):
    testNum = len(testSet)
    rightNum = 0.0
    for index in range(testNum):
        [tempLabel, softmax] = forward_test(testSet[index], testLabel[index])
        print 'Real Label:'
        print testLabel[index]
        print 'Predict Label'
        print tempLabel
        
        if tempLabel == testLabel[index]:
            rightNum = rightNum + 1
    accuRate =  rightNum / testNum
    print u'测试集准确率: '
    print accuRate
    return accuRate
    
if __name__=="__main__":
    fileListName = 'iris.txt'
    trainSet = []
    testSet = []
    trainLabel = []
    testLabel = []
    [totalSet,totalNum,totalLabel] = readAllFile(fileListName)
#    totalNum = 150
    anchor = splitTrainTest(totalNum)
    for index in range(len(anchor)):
        if anchor[index] == 1:
            trainSet.append(totalSet[index])
            trainLabel.append(totalLabel[index])
        else:
            testSet.append(totalSet[index])
            testLabel.append(totalLabel[index])
    print u'训练集大小: '
    print len(trainSet)
    print u'测试集大小: '
    print len(testSet)
    
    train(trainSet, trainLabel)
    accuRate = test(testSet, testLabel)
    
 
