from TFRecord import get_File,float32_feature,bytes_feature
from skimage import io
from utilities.log import TensorFlow_log
import tensorflow as tf
import numpy as np

# global value.
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
IMAGE_DEPTH = 3
one_hot_depth = 4


if __name__ =='__main__':
    # try:
    #     image = io.imread("image.jpg")
    #     if image is None:
    #         TensorFlow_log.warning('Error images')
    #     else:
    #         image_raw = image.tostring()
    #     # check the image shape.        
    #     image_byte = bytes_feature(image_raw)
    #     image_content = tf.decode_raw(image_byte, tf.uint8)
    #     image = tf.reshape(image_content, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
    #     img = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    # except:
    #     # image is not in this folder.
    #     TensorFlow_log.error('Input prediction image failed.')
      
    
    # setting placeholder.
    # tf_x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH])/255
    # image = tf.reshape(tf_x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])     # (batch, height, width, channel)
    # tf_y = tf.placeholder(tf.float32, [None,one_hot_depth])
    # Add ops to save and restore all the variables.
    # saver = tf.train.Saver()
    saver = tf.train.import_meta_graph('./Model/CNN.model-199.meta')
    # turn on tensorflow.
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    graph = tf.get_default_graph()
    w1 = graph.get_tensor_by_name('tf_x:0')
    output = graph.get_tensor_by_name('output:0')
    # set test dict.
    saver.restore(sess,tf.train.latest_checkpoint('./Model'))
    print(output)

    # test_feature = sess.run(img)



