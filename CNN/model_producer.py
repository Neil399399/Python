from TFRecord import get_File,TFRecord_Writer,TFRecord_Reader
from utilities.log import TensorFlow_log
import tensorflow as tf
import numpy as np
# global value.
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
IMAGE_DEPTH = 3
one_hot_depth = 5
LR = 0.001


if __name__ =='__main__':
    # train data.
    train_images,train_labels = TFRecord_Reader('./TFRecord/train.tfrecord',IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,30)
    test0_images,test0_labels = TFRecord_Reader('./TFRecord/test.tfrecord0',IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,20)


    # setting placeholder.
    tf_x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH],name='tf_x')/255
    image = tf.reshape(tf_x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])              # (batch, height, width, channel)
    tf_y = tf.placeholder(tf.float32, [None,one_hot_depth])
    
    # CNN.
    output_shape = (IMAGE_HEIGHT/2**2)*(IMAGE_WIDTH/2**2)*36
    # (image_height, image_width, Conv1_Filter)
    conv1 = tf.layers.conv2d(inputs=image, filters=6, kernel_size=5, strides=1, padding='same', activation=tf.nn.relu,name='conv1')
    pool1 = tf.layers.max_pooling2d(conv1,pool_size=2,strides=2,name='pool1')  # -> (image_height/Pool_Size, image_width/Pool_Size, Conv1_Filter)
    # (image_height, image_width, Conv2_Filter)
    conv2 = tf.layers.conv2d(pool1, 36, 5, 1, 'same', activation=tf.nn.relu,name='conv2')    
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2,name='pool2')
    # flat.
    flat = tf.reshape(pool2, [-1, int(output_shape)])
    # output layer.
    output = tf.layers.dense(flat, one_hot_depth,name='output')


    # def loss, accuracy.
    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)
    accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

    # turn on tensorflow.
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # open queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    sess.run(init_op)

    # set train dict.
    train_feature, train_label = sess.run([train_images,train_labels])
    # decode train_label to one_hot.
    train_label_onehot = sess.run(tf.one_hot(train_label,one_hot_depth))

    # set test dict.
    test0_feature, test0_label = sess.run([test0_images,test0_labels])
    # decode test_label to one_hot.
    test0_label_onehot = sess.run(tf.one_hot(test0_label,one_hot_depth))

    # saver.
    saver = tf.train.Saver(max_to_keep=1)

    # training.
    TensorFlow_log.info('Make graph and start trainng.')
    for step in range(200):
        TensorFlow_log.info('Training step :%d',step)
        _, loss_ = sess.run([train_op, loss], {tf_x: train_feature, tf_y: train_label_onehot})
        if step % 10 == 0:
            validate_accuracy = sess.run(accuracy,{tf_x: train_feature, tf_y: train_label_onehot})
            TensorFlow_log.info('After %d training step(s), the validation accuracy is %.2f.',step,validate_accuracy)
            TensorFlow_log.info('loss : %s',loss_)

    # final validate with used test data.
    TensorFlow_log.info('Start Testing.')

    test0_accuracy = sess.run(accuracy,{tf_x: test0_feature, tf_y: test0_label_onehot})
    TensorFlow_log.info('Test0 accuracy : %.2f',test0_accuracy)

    # stop all threads
    coord.request_stop()
    coord.join(threads)
    TensorFlow_log.info('CNN training done.')

    # save model.
    # saver.save(sess,'./Model/CNN.model',global_step=step)
