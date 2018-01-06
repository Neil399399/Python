import numpy as np
import csv

Data=[]
Index=[]

#open original data
T= open('/home/wril/Documents/Python-TensorFlow/IP-Train/Lablog.csv', 'r')
for data in csv.reader(T,delimiter=','):
    Data.append(data)
  
#open original index
t= open('/home/wril/Documents/Python-TensorFlow/IP-Train/Collegeindex.csv', 'r') 
for index in csv.reader(t,delimiter=','):
    Index.append(index)
 
#change data college to index number
for i in range(0,len(Data)):
    for j in range(0,len(Index)):
        if Data[i][0] == Index[j][1]:
           Data[i][0] = Index[j][0]

#print result
# for i in range(0,len(Data)):
#       print(Data[i][1])
# print(len(Data))

#write in file 
F= open('/home/wril/Documents/Python-TensorFlow/IP-Train/TrainData.csv', 'wb')
w = csv.writer(F)
w.writerows(Data)
F.close()
               