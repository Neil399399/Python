from utilities.log import TFRecord_log
from skimage import io
from shutil import copy
import tensorflow as tf
import numpy as np
import os 

# 二進位資料
def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 整數資料
def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 浮點數資料
def float32_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def get_File(file_dir):
  # The images in each subfolder
  images = []
  # The subfolders
  subfolders = []
  # Using "os.walk" function to grab all the files in each folder
  for dirPath, dirNames, fileNames in os.walk(file_dir):
    for name in dirNames:
      subfolders.append(os.path.join(dirPath, name))
    
  for folder in subfolders:
    for dirPath, dirNames, fileNames in os.walk(folder):
      for image_name in fileNames:
        images.append(image_name)
        # copy file to make training data.
        copy(os.path.join(dirPath, image_name),'./train/')
  
  # To record the labels of the image dataset. ex: [0,0,1,1,2,2,2]
  labels = []
  count = 0
  for a_folder in subfolders:
      n_img = len(os.listdir(a_folder))
      TFRecord_log.info('label - folder : %s %d',a_folder,count)
      labels = np.append(labels, n_img * [count])
      count+=1

  # merge label and feature.
  subfolders = np.array([images, labels])
  subfolders = subfolders[:, np.random.permutation(subfolders.shape[1])].T

  image_list = list(subfolders[:, 0])
  label_list = list(subfolders[:, 1])
  label_list = [int(float(i)) for i in label_list]
  return image_list, label_list

def TFRecord_Writer(images, labels, images_dir,image_folder, TFrecord_name):
  n_samples = len(labels)
  TFWriter = tf.python_io.TFRecordWriter(TFrecord_name)
  TFRecord_log.info('Start make TFRecord file.')
  for i in np.arange(0, n_samples):
    try:
      image = io.imread(images_dir+image_folder+'/'+images[i])
      if image is None:
        TFRecord_log.warning('Error image:' + images[i])
      else:
        image_raw = image.tostring()

      label = int(labels[i])
      height, width, depth = image.shape
      # check the image shape.
      if height != 640 or width !=640:
            continue
      # take tf.train.Feature and merge to tf.train.Features.
      ftrs = tf.train.Features(feature={'Label': int64_feature(label),'image_raw': bytes_feature(image_raw),
                                'height':int64_feature(height),'width': int64_feature(width)})
      # take tf.train.Features and change to tf.train.Example.
      example = tf.train.Example(features=ftrs)
      # take tf.train.Example and write in tfRecord file.
      TFWriter.write(example.SerializeToString())
    except:
      # image is not in this folder.
      continue

  TFWriter.close()
  TFRecord_log.info('Make TFRecord file done.')

def TFRecord_Reader(TFRecord_Files,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,Batch_Size):
    TFRecord_log.info('Start read TFRecord file.')
    # create queue.
    try:
      filename_queue = tf.train.string_input_producer(TFRecord_Files,shuffle=True,num_epochs=None)
    except:
      TFRecord_log.error('Input data in queue faild !!')
    # reader.
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # get features.
    img_features = tf.parse_single_example(serialized_example,features={
                                    'Label'    : tf.FixedLenFeature([], tf.int64),
                                    'image_raw': tf.FixedLenFeature([], tf.string),
                                    'height': tf.FixedLenFeature([], tf.int64),
                                    'width': tf.FixedLenFeature([], tf.int64), })
    
    # recover image. 
    TFRecord_log.info('Reshape image.')
    try:
      image_content = tf.decode_raw(img_features['image_raw'], tf.uint8)
      # image_float32 = tf.image.convert_image_dtype(image_content,tf.float32)
      image = tf.reshape(image_content, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
      img = tf.cast(image, tf.float32) * (1. / 255) - 0.5
      label = tf.cast(img_features['Label'], tf.float32)
    except:
      TFRecord_log.error('Reshape image failed !!')
    # regulate images size.
    resized_image = tf.image.resize_image_with_crop_or_pad(image=img,target_height=IMAGE_HEIGHT,target_width=IMAGE_WIDTH)
    images, labels = tf.train.shuffle_batch(
                            [resized_image, label],
                            batch_size= Batch_Size,
                            capacity=10000+3*Batch_Size,
                            min_after_dequeue=1000)
    return images, labels