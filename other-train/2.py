import tensorflow as tf
import numpy as np
import csv

with open('test(only one 1).csv', 'r') as csvfile:
     Train = [list(map(int,rec)) for rec in csv.reader(csvfile, delimiter=',')]


 #input xs



with open('test2(only one 1).csv', 'r') as csvfile:
     
     Test = [list(map(int,rec)) for rec in csv.reader(csvfile, delimiter=',')]
  

def add_layer(inputs, in_size, out_size, activation_function=None,):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
    return outputs
def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# define placeholder for inputs to network

xs = tf.placeholder(tf.float32, [None, 3]) # 28x28
ys = tf.placeholder(tf.float32, [None, 3])



# add output layer

prediction = add_layer(xs, 3, 3,  activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
sess.run(tf.global_variables_initializer())
 
print Train[2]

for i in range(1000):
    
    
    batch_xs = Train

    batch_ys = Test

    sess.run(train_step ,feed_dict={xs:batch_xs, ys: batch_ys})
    if i % 50 == 0:
        
        print(compute_accuracy(
             batch_xs,batch_ys))