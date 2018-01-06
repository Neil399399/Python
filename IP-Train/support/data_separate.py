#separate train data to two classification: train data and train label
import numpy as np
import csv

origonalData=[]
trainData=[]
trainLabel=[]

#open file
# T= open('/home/wril/Documents/Python-TensorFlow/IP-Train/TrainData.csv', 'r')
# for data in csv.reader(T,delimiter=','):
#     origonalData.append(data)
  
# for i in range(1,len(origonalData)):
#     label = origonalData[i][0]
#     #transfer datatype to int
#     #function isinstance : check the data type -> return bool
#     #print isinstance(int(label),int)
#     trainLabel.append(int(label))

# #print(trainLabel)


# for i in range(1,len(origonalData)):
#     data = origonalData[i][1]
#     temp=[]
#     #split IP [111.111.111.111] => [111 111 111 111]
#     a = data.split('.')
#     #transfer datatype to int and append to temp[]
#     temp.append(int(a[0]))
#     temp.append(int(a[1]))
#     temp.append(int(a[2]))
#     temp.append(int(a[3]))
#     #temp[] writeback to trainData[]
#     trainData.append(temp)
# #print(trainData)


F = open('/home/wril/Documents/Python-TensorFlow/IP-Train/merge.csv','r')
for data in csv.reader(F,delimiter=','):
        origonalData.append(data)

for i in range(0,len(origonalData)):
    label = origonalData[i][0]
    trainLabel.append(int(label))

for i in range(0,len(origonalData)):
    temp=[]
    for j in range(1,len(origonalData[i])):
        data = origonalData[i][j]        
        temp.append(float(data))
    trainData.append(temp)

