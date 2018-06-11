from TFRecord import get_File,TFRecord_Writer,TFRecord_Reader
from cnn import CNN_Model
from utilities.log import TFRecord_log
import tensorflow as tf
import numpy as np
import os
import shutil

# global value.
image_Dir = './example_data/'
image_folder_list = ['台灣','早餐','美食','捷運']
tfrecord_files_path = ['./TFRecord/test.tfrecord0','./TFRecord/test.tfrecord1','./TFRecord/test.tfrecord2','./TFRecord/test.tfrecord3']
# ['台灣','美食','捷運','早餐']

  
if __name__ =='__main__':
   # make image to .TFRecord file.
    i=0
    if not os.path.exists('./train'):
        os.makedirs('./train')
    image_list,label_list = get_File(image_Dir)
    # make train tfrecord.
    TFRecord_Writer(image_list,label_list,'./','train','./TFRecord/train.tfrecord')
    shutil.rmtree('./train', ignore_errors=True)
    TFRecord_log.info('Remove train folder.')
    # make test tfrecord.
    for each_folder in image_folder_list:
        TFRecord_Writer(image_list,label_list,image_Dir,each_folder,'./TFRecord/test.tfrecord'+str(i))
        i+=1

    # train_images,train_labels = TFRecord_Reader(['./TFRecord/train.tfrecord'],640,640,3,5)
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