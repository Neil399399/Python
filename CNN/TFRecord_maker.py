from TFRecord import get_File,TFRecord_Writer,TFRecord_Reader
from cnn import CNN_Model
from utilities.log import TFRecord_log
import tensorflow as tf
import numpy as np
import os
import shutil

# global value.
train_Dir = './train_data/'
test_Dir = './test_data/'
tfrecord_dir = './TFRecord/'

  
if __name__ =='__main__':
   # make image to .TFRecord file.
    # make folder if not exsit.
    if not os.path.exists('./train'):
        os.makedirs('./train')
    if not os.path.exists('./test'):
        os.makedirs('./test')
    if not os.path.exists('./TFRecord'):
        os.makedirs('./TFRecord')

    train_image_list,train_label_list, folders = get_File(train_Dir)
    test_image_list,test_label_list, folders = get_File(test_Dir)

    # make train tfrecord.
    TFRecord_Writer(train_image_list,train_label_list,'./','train',tfrecord_dir,'train.tfrecord')
    shutil.rmtree('./train', ignore_errors=True)
    TFRecord_log.info('Remove train folder.')
    # make test tfrecord.
    TFRecord_Writer(test_image_list,test_label_list,'./','test',tfrecord_dir,'test.tfrecord')
    shutil.rmtree('./test', ignore_errors=True)
    TFRecord_log.info('Remove test folder.')
