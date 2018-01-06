'''
Created on Apr 29, 2017

@author: eric
'''

from __future__ import print_function
import tensorflow as tf
import time
import numpy as np
#tf.device("/cpu:0")
row_size = 902
col_size = 200
batch_size = 100
recordTime = time.time()

def add_layer(inputs, in_size, out_size, layer_name, b,activation_function=None ):
    # add one more layer and return the output of this layer
     with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + b)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

def compute_accuracy(test_datas, test_labels):
    global prediction
    y_pre = sess.run(prediction, feed_dict={train_datas: test_datas})
    v_ys_tab = sess.run(tf.argmax(test_labels, 1))
    totle = 0
    for i in range(len(y_pre)):
        totle += rank(y_pre[i],v_ys_tab[i])
    return round(totle * 100 / (i + 1.0) / len(y_pre[0]), 2)
def rank(x,i):
    p=x[i]
    ranks=0
    for i in range(len(x)):
        if(x[i]<p):
            ranks +=1
    
    return ranks
def gettime():
    global recordTime
    tempTime = recordTime
    recordTime = time.time()
    return round(recordTime - tempTime, 2)

data = np.reshape(np.genfromtxt('test_data(902*200).csv', delimiter=','), (row_size, col_size))
label = np.reshape(np.genfromtxt('test_label(902*200).csv', delimiter=','), (row_size, col_size))

with tf.Graph().as_default():
    # inputs
    xs = tf.placeholder(dtype=tf.float32, shape=[row_size, col_size],name="train_data_input")
    ys = tf.placeholder(dtype=tf.float32, shape=[row_size, col_size],name="train_label_input")
    #xs1 = tf.placeholder(tf.float32, [None, 200])
    #ys1= tf.placeholder(tf.float32, [None, 200])
    
    v_xs = tf.Variable(xs, trainable=False, collections=[])
    v_ys = tf.Variable(ys, trainable=False, collections=[])
    (train_data, train_label,) = tf.train.slice_input_producer([v_xs, v_ys])
    train_datas, train_labels = tf.train.batch([train_data, train_label], batch_size=batch_size,name="train_batch")
    #test_datas, test_labels = tf.train.batch([train_data, train_label], batch_size=batch_size,name="test_batch")
    # layer
    input_layer = add_layer(train_datas,200,200, 'input_layer', 0.1,activation_function=None)
    hidden_layer1 = add_layer(input_layer,200,300, 'hidden_layer1', 0.1,activation_function=tf.nn.relu)
    hidden_layer2 = add_layer(hidden_layer1 ,300,400,'hidden_layer2',0.3, activation_function=None)
    prediction = add_layer(hidden_layer2, 400, 200,'output_layer',0.3,activation_function=tf.nn.softsign)
    
    # loss
    with tf.name_scope('loss'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(train_labels * tf.log((prediction +1)/2+ 1e-15),
                                                  reduction_indices=[1]))
    #summary
    #tf.summary.scalar('loss', cross_entropy)
    
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    tStart = time.time()
    with tf.Session() as sess:
        #------------------------------------------------------------------#
        merged  = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("logs/train", sess.graph)
        test_writer = tf.summary.FileWriter("logs/test", sess.graph)
        #--------------------------------------------------------------------#
        sess.run(tf.global_variables_initializer())
        sess.run(v_xs.initializer, feed_dict={xs: data})
        sess.run(v_ys.initializer, feed_dict={ys: label})
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            step = 0
            while not coord.should_stop():
                sess.run(train_step)
                
                if step % 500 == 0:
                    _test_datas, _test_labels = sess.run([train_datas, train_labels ])

                    #print(step, gettime())
                    print(step, compute_accuracy(_test_datas, _test_labels), gettime())
                   # print(sess.run(cross_entropy))
                     # summary of train aand tst   
                    #train_result = sess.run(merged, feed_dict={xs1:_test_datas,ys1:_test_labels})
                    #test_result = sess.run(merged, feed_dict={xs1:_test_datas,ys1:_test_labels})
                    #train_writer.add_summary(train_result, step)
                    #test_writer.add_summary(test_result, step)
                    
                    tEnd2 = time.time()
                    print("complete, time %f sec"%(tEnd2-tStart))
                    print("loss", sess.run(cross_entropy))   
                if step == 10000:
                   
                    i = 0
                    j=0
                    while j<10:
                        _test_datas, _test_labels = sess.run([train_datas, train_labels])
                        i += compute_accuracy(_test_datas, _test_labels)
                        j+=1
                    print("Average accuracy:",i/10,"%")
                    
                    coord.request_stop()
                step = step + 1
        except tf.errors.OutOfRangeError:
            print('Saving')
        finally:
            coord.request_stop()
    coord.join(threads)
    