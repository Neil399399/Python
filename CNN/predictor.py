from TFRecord import bytes_feature
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
    TensorFlow_log.info('Start predict.')
    TensorFlow_log.info('load images.')
    try:
        image = io.imread("./example_data/new-7.jpg")
        if image is None:
            TensorFlow_log.warning('Error images')
        else:
            image_raw = image.tostring()
        # check the image shape.
        image_byte = bytes_feature(image_raw)
        image_content = tf.decode_raw([image_raw], tf.uint8)
        image = tf.reshape(image_content, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
        img = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    except:
        # image is not in this folder.
        TensorFlow_log.error('Input prediction image failed.')
      
    # load graph.
    saver = tf.train.import_meta_graph('./Model/CNN.model-199.meta')
    # load model.
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # load input,output value.
    graph = tf.get_default_graph()
    tf_x = graph.get_tensor_by_name('tf_x:0')
    saver.restore(sess,tf.train.latest_checkpoint('./Model/'))
    output = graph.get_tensor_by_name('output/MatMul:0')

    # initialize input value.
    input = sess.run([img])
    test_output = sess.run(output, {tf_x: input})
    prediction = np.argmax(test_output, 1)
    print('Prediction:',prediction)
 



