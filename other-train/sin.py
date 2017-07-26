import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def add_layer(inputs, in_size, out_size,n_layer, activation_function=None):
    #-------add one more layer and return the output of this layer----------#
  layer_name='layer%s'%n_layer
  with tf.name_scope(layer_name):              # using in tf.graph
    with tf.name_scope('weights'):
      Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
      tf.histogram_summary(layer_name+'/weights',Weights)       #using in jistory graph
    with tf.name_scope('biases'):
      biases = tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
      tf.histogram_summary(layer_name+'/biases',biases)
    with tf.name_scope('Wx_plus_b'):
      Wx_plus_b = tf.matmul(inputs,Weights)+biases
    
    if activation_function is None:
        outputs = Wx_plus_b
    else:
          outputs = activation_function(Wx_plus_b,)
    tf.histogram_summary(layer_name+'/outputs',outputs)
    
    return outputs
        
 #----------------create data-----------------------#
c1 = np.array([4,0])#[:, np.newaxis]
c2 = np.array([0,20])#[:, np.newaxis]
x1_data = np.linspace(-3,12.1,10,True,dtype=np.float32)#[:, np.newaxis]     # -3 ~ 12.1 ,has 1000 rows
x2_data = np.linspace(4.1,5.8,10,True,dtype=np.float32)#[:, np.newaxis]     # 4.1 ~ 58 ,has 1000 rows
x_data = np.shape((x1_data,x2_data))
noise = np.random.normal(0,0.5,x1_data.shape).astype(np.float32)     # noise variance = 0.05

y_data = 21.5+np.sin(4*x1_data*np.pi)*x1_data+np.sin(20*np.pi*x2_data)*x2_data              # x*x - 0.5 + noise

#----------------define placeholder for inputs to network ----------------#
with tf.name_scope('inputs'):
     xs = tf.placeholder(tf.float32,[None,2],name= 'x_input')
     ys = tf.placeholder(tf.float32,[None,1],name= 'y_input')

#----------------add hidden layer-----------------------#
#L1 = add_layer(xs,2,10,n_layer=1,activation_function=None)    # add hide layer ,out_size=10
#----------------add output layer-----------------------#
prediction = add_layer(xs,2,1,n_layer=2,activation_function =tf.nn.relu)     # add hide layer ,in_size=L1


#-------the error between predicition and real data------------#

with tf.name_scope('loss'):
  loss =tf.reduce_mean(tf.reduce_sum(tf.square(prediction-ys),reduction_indices=[1]))    #  mean(sum(loss = (real-pre)*2))
  tf.scalar_summary('loss',loss)    #using in look events of loss


with tf.name_scope('train'):
  train_step =tf.train.GradientDescentOptimizer(0.1).minimize(loss)    # learning_rate = 0.1

#--------------initialize all variables--------------#
init = tf.initialize_all_variables()

sess = tf.Session()
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("logs/",sess.graph)
#-------important step------------#
sess.run(init)

#------------------plot result---------------#
#fig = plt.figure()
#ax = Axes3D(fig)
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
#plt.ion()
#plt.show()

#-------------------run---------------------#

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    
    if i % 10==0:
       # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        print(sess.run(prediction,feed_dict={xs:x_data}))
        print(x_data)
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)
        #----------------------print line of the every results------------------#
        #try:
         #  ax.lines.remove(lines[0])
       # except Exception:
        #    pass
        #prediction_value = sess.run(prediction,feed_dict = {xs:x_data})
        #lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        #plt.pause(0.2)
        