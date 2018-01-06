import numpy as np
import csv

with open('imdbtrain.csv', 'r') as csvfile:
     Train = [list(map(float,rec)) for rec in csv.reader(csvfile, delimiter=',')]
  


n=0
for a in range (0,len(Train)):
  
  M= max(Train[a])
  if M==0:
         n=n+1
          
print n
for a in range (0,len(Train)-n):
          
  M= max(Train[a])
  if M==0:
         del Train[a]
          
         
     
F= open('train.csv', 'wb')
w = csv.writer(F)
w.writerows(Train)
F.close()
               
       




               