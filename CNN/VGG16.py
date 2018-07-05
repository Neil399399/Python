import tensorflow as tf
import numpy as np
from cnn import Vgg16,Train
from skimage import io
from TFRecord import get_File,TFRecord_Writer,TFRecord_Reader
from utilities.log import TensorFlow_log
from utilities.util import F1_Score,Precision,Recall

# global value.
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
IMAGE_DEPTH = 3
one_hot_depth = 6

if __name__ =='__main__':
    # train data.
    train_images,train_labels = TFRecord_Reader('./TFRecord/train.tfrecord',IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,50)
    test_images,test_labels = TFRecord_Reader('./TFRecord/test.tfrecord',IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,30)

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
    test_feature, test_label = sess.run([test_images,test_labels])
    # decode test_label to one_hot.
    test_label_onehot = sess.run(tf.one_hot(test_label,one_hot_depth))


    TensorFlow_log.info('Make graph and start training.')
    Train(train_feature,train_label_onehot,test_feature,test_label_onehot)
    # close queue.
    coord.request_stop()
    coord.join(threads)
    TensorFlow_log.info('CNN training done.')