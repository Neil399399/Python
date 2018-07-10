import csv

def Accuracy(predictions,labels):
    count = 0
    for index in range(len(predictions)):
        if predictions[index]==labels[index]:
            count +=1
    return count/len(predictions)

def Precision(predictions,labels,label):
    TP = 0
    FP = 0
    try:
        for index in range(len(predictions)):
            if predictions[index]==label:
                if labels[index]==label:
                    TP+=1
                else:
                    FP+=1
        return TP/(TP+FP)
    except:
        return 0


def Recall(predictions,labels,label):
    TP = 0
    FN = 0
    try:
        for index in range(len(labels)):
            if labels[index]==label:
                if predictions[index]==label:
                    TP+=1
                else:
                    FN+=1
        return TP/(TP+FN)
    except:
        return 0



def F1_Score(precision,recall):
    if precision == 0 or recall == 0:
        return 0
    else:
        return (2*precision*recall/(precision+recall))


def Writer(file,input):
    with open(file,'a',newline='') as csvfile:
        writer = csv.writer(csvfile,delimiter=',')
        writer.writerow(input)