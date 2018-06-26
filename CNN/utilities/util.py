def Precision(predictions,labels,label):
    TP = 0
    FP = 0
    for index in range(len(predictions)):
        if predictions[index]==label:
            if labels[index]==label:
                TP+=1
            else:
                FP+=1
    return TP/(TP+FP)


def Recall(predictions,labels,label):
    TP = 0
    FN = 0
    for index in range(len(labels)):
        if labels[index]==label:
            if predictions[index]==label:
                TP+=1
            else:
                FN+=1
    return TP/(TP+FN)



def F1_Score(precision,recall):
    return (2*precision*recall/(precision+recall))