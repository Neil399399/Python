from TFRecord import get_File,TFRecord_Writer,TFRecord_Reader
from cnn import CNN_Model
from utilities.log import TFRecord_log
import tensorflow as tf
import numpy as np
image_Dir = './example_data/'
image_folder_list = ['台灣','美食','捷運','早餐']


  
if __name__ =='__main__':
   # make image to .TFRecord file.
    image_list,label_list = get_File(image_Dir)
    TFRecord_Writer(image_list,label_list,image_Dir,image_folder_list,'test.tfrecords')
    # train_images,train_labels = TFRecord_Reader('test.tfrecords',640,640,3,10)
    # # turn on tensorflow.
    # sess = tf.Session()
    # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # # open queue.
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    # sess.run(init_op)

    # # set train dict.
    # for i in range(5):
    #     train_feature, train_label = sess.run([train_images,train_labels])
    #     print(train_feature[1])
    #     print(train_label)