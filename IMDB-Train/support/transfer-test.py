import numpy as np
import csv

with open('train.csv', 'r') as csvfile:
     Train = [list(map(float,rec)) for rec in csv.reader(csvfile, delimiter=',')]



for a in range (0,len(Train)):
   count=0
   M= max(Train[a])
   
   for n,i in enumerate(Train[a]):
      if i==M:
          if count==0:
               Train[a][n]=1.0
               count=count+1
               
          else:
             Train[a][n]=0.0
      else:
         Train[a][n]=0.0
     
     
            
 
F= open('imdbtest.csv', 'w') 
w = csv.writer(F)
w.writerows(Train)
F.close()
               
               
    
