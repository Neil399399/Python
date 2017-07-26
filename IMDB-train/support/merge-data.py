import numpy as np
import csv

T= open('test.csv', 'r')
Test = [list(map(float,rec)) for rec in csv.reader(T, delimiter=',')]
  
t= open('train.csv', 'r') 
Train = [list(map(float,rec)) for rec in csv.reader(t, delimiter=',')]
  

for a in range(0,len(Train)):
         Train[a].extend(Test[a])



F= open('merge.csv', 'wb')
w = csv.writer(F)
w.writerows(Train)
F.close()
               