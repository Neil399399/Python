import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from pylab import mpl
import csv

#weight and biases
#--------------------------------------------------------------------------------------#
def add_layer(inputs, in_size, out_size, layer_name, b,activation_function=None ):
    # add one more layer and return the output of this layer
     with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal(shape=[in_size, out_size]))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros(shape=[1, out_size]) + b)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs,Weights


#####Data Parse#####
origonalData=[]
trainData=[]
trainLabel=[]

#open file
T= open('/home/wril/Documents/Python-TensorFlow/IP-Train/TrainData.csv', 'r')
for data in csv.reader(T,delimiter=','):
    origonalData.append(data)
  
for i in range(1,len(origonalData)):
    label = origonalData[i][0]
    trainLabel.append(int(label))

for i in range(1,len(origonalData)):
    data = origonalData[i][1]
    temp=[]
    #split IP [111.111.111.111] => [111 111 111 111]
    a = data.split('.')
    #transfer datatype to int and append to temp[]
    temp.append(int(a[0]))
    temp.append(int(a[1]))
    temp.append(int(a[2]))
    temp.append(int(a[3]))
    #temp[] writeback to trainData[]
    trainData.append(temp)
#print(trainData)

# F = open('/home/wril/Documents/Python-TensorFlow/IP-Train/merge.csv','r')
# for data in csv.reader(F,delimiter=','):
#         origonalData.append(data)

# for i in range(0,len(origonalData)):
#     label = origonalData[i][0]
#     trainLabel.append(int(label))

# for i in range(0,len(origonalData)):
#     temp=[]
#     for j in range(1,len(origonalData[i])):
#         data = origonalData[i][j]        
#         temp.append(float(data))
#     trainData.append(temp)


######model######

#Initalization tensorflow
#--------------------------------------------------------------------------#
mpl.rcParams['font.sans-serif'] = ['SimHei']
np.random.seed(1)
tf.set_random_seed(1)
sess=tf.Session()

#input data
#--------------------------------------------------------------------------#
#iris=datasets.load_iris()
#x_vals= iris.data
x_vals= np.array(trainData)
target= trainLabel
#target= iris.target
y_vals=np.array([1 if y==0 else -1 for y in target])

#setting train and test data
#-------------------------------------------------------------------------------------#
#x_vals_train = [111 111 111 1111]   y_vals_train = [label]
train_indices = np.random.choice(len(x_vals),round(len(x_vals)*0.8),replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

data_train = x_vals[train_indices]
data_test = x_vals[test_indices]
label_train = y_vals[train_indices]
label_test = y_vals[test_indices]

#setting train data format
#--------------------------------------------------------------------------------------#
batch_size = 100
x_data = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)


#use L2 normalization
#alpha is constant (use in loss)
#Layer
#-------------------------------------------------------------------------------------#
layer1,Weights = add_layer(x_data, 4, 8,'layer1',0,activation_function=None)
output,Weights = add_layer(layer1, 8, 1,'output_layer',0,activation_function=None)

#--------------------------------------------------------------------------------------#
#model_output=tf.matmul(x_data,W)+b
l2_norm = tf.reduce_sum(tf.square(Weights))
alpha = tf.constant([0.1])
#setting loss
classification_term = tf.reduce_mean(tf.maximum(0.,1.-output*y_target))
loss = classification_term+alpha*l2_norm
#prediction
prediction = tf.sign(output)
#accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target),tf.float32)+ 1e-15)
#train_step
with tf.name_scope('train'):
     train_step=tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#Initalization training 
#-------------------------------------------------------------------------------------#
sess.run(tf.global_variables_initializer())
loss_vec = []
train_accuracy = []
test_accuracy = []

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)
#setting train

for i in range(100):
    rand_index = np.random.choice(len(data_train), size=batch_size)
    rand_x = data_train[rand_index]
    rand_y = np.transpose([label_train[rand_index]])
    #run
    sess.run(train_step, feed_dict={x_data: rand_x, y_target:rand_y})
    #Record every loss
    temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
    loss_vec.append(temp_loss)
    #Record every train accuracy
    train_acc_temp = sess.run(accuracy, feed_dict={x_data: data_train, y_target: np.transpose([label_train])})
    train_accuracy.append(train_acc_temp)
    #Record every test accuracy
    test_acc_temp = sess.run(accuracy, feed_dict={x_data: data_test, y_target: np.transpose([label_test])})
    test_accuracy.append(test_acc_temp)
    #every run n time, output accuracy and loss.
    if (i+1)%10==0:
        print('Step #' + str(i+1))
        print('Loss = ' + str(temp_loss))
        print('accuracy = ' + str(train_acc_temp))
        

#matlabplot
plt.plot(loss_vec)
plt.plot(train_accuracy)
plt.plot(test_accuracy)
plt.legend(['loss','train accuracy','test accuracy'])
plt.ylim(0.,1.)
plt.show()