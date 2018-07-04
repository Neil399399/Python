from TFRecord import bytes_feature
from skimage import io
from utilities.util import Writer
from utilities.log import TensorFlow_log
import tensorflow as tf
import numpy as np
import os 

# global value.
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
IMAGE_DEPTH = 3
one_hot_depth = 6
file_dir = './example_data'
output_file = 'test.csv'

if __name__ =='__main__':
    
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
    
    
    TensorFlow_log.info('Load images.')
    # get images from folder.
    # Using "os.walk" function to grab all the files in each folder
    for dirPath, dirNames, fileNames in os.walk(file_dir):
        for name in dirNames:
            subfolder_path = os.path.join(dirPath, name)
            label_0 = 0
            label_1 = 0
            label_2 = 0
            label_3 = 0
            label_4 = 0
            label_5 = 0
            for dirPath, dirNames, fileNames in os.walk(subfolder_path):
                imgs = []
                for image_name in fileNames:
                    # open image.
                    try:
                        image = io.imread(os.path.join(dirPath, image_name))
                        if image is None:
                            TensorFlow_log.warning('Error images')
                        else:
                            image_raw = image.tostring()
                        # check the image shape.
                        image_byte = bytes_feature(image_raw)
                        image_content = tf.decode_raw([image_raw], tf.uint8)
                        image = tf.reshape(image_content, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
                        img = tf.cast(image, tf.float32) * (1. / 255) - 0.5
                        imgs.append(img)
                    except:
                    # image is not in this folder.
                        TensorFlow_log.error('Input prediction image failed.')


                # initialize input value.
                input = sess.run(imgs)
                # start predict.
                TensorFlow_log.info('Start predict user %s images ...',name)
                test_output = sess.run(output, {tf_x: input})
                prediction = np.argmax(test_output, 1)
                TensorFlow_log.info('Finish prediction.')
                TensorFlow_log.info('Save in file.')
                for value in prediction:
                    if value == 0:
                        label_0+=1
                    if value == 1:
                        label_1+=1
                    if value == 2:
                        label_2+=1
                    if value == 3:
                        label_3+=1
                    if value == 4:
                        label_4+=1
                    if value == 5:
                        label_5+=1
            # result
            result = [name,label_0,label_1,label_2,label_3,label_4,label_5]
            Writer(output_file,result)
    TensorFlow_log.info('Finish all user predict and save.')

            
 



