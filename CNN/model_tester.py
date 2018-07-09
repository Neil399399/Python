from TFRecord import bytes_feature
from skimage import io
from utilities.util import Writer, Precision, Recall, F1_Score
from utilities.log import TensorFlow_log
import tensorflow as tf
import numpy as np
import os 
import gc

# global value.
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
IMAGE_DEPTH = 3
one_hot_depth = 6
file_dir = './example'
output_file = 'test.csv'
label = 3
if __name__ =='__main__':
    

    # load graph.
    saver = tf.train.import_meta_graph('./Model/CNN.model-200.meta')
    # load model.
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # load input,output value.
    graph = tf.get_default_graph()
    tf_x = graph.get_tensor_by_name('Input/tf_x:0')
    output = graph.get_tensor_by_name('output/MatMul:0')
    
    TensorFlow_log.info('Load images.')
    # get images from folder.
    # Using "os.walk" function to grab all the files in each folder
    for dirPath_root, dirNames, fileNames in os.walk(file_dir):
        for name in dirNames:
            subfolder_path = os.path.join(dirPath_root, name)
            sess = tf.Session()
            saver.restore(sess,tf.train.latest_checkpoint('./Model/'))

            for dirPath, dirNames, fileNames in os.walk(subfolder_path):
                imgs = []
                labels = []
                if len(fileNames)==0:
                    continue
                print(fileNames)
                for image_name in fileNames:
                    # open image.
                    if len(imgs)<50:
                        try:
                            image = io.imread(os.path.join(dirPath, image_name))
                            height, width, depth = image.shape
                            if image is None:
                                TensorFlow_log.warning('Error images')
                            elif height == 640 or width ==640:
                                image_raw = image.tostring()
                            else:
                                TensorFlow_log.warning('Image has wrong size.')
                            # check the image shape.
                            image_byte = bytes_feature(image_raw)
                            image_content = tf.decode_raw([image_raw], tf.uint8)
                            image = tf.reshape(image_content, [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])
                            img = tf.cast(image, tf.float32) * (1. / 255) - 0.5
                            imgs.append(img)
                            # labels
                            labels.append(label)
                        except:
                            # image is not in this folder.
                            TensorFlow_log.error('Input prediction image failed.')
                    else:
                        break
                # initialize input value.
                input = sess.run(imgs)
                # start predict.
                TensorFlow_log.info('Start predict user %s images ...',name)
                test_output = sess.run(output, {tf_x: input})
                prediction = np.argmax(test_output, 1)
                print(prediction)
                TensorFlow_log.info('Finish prediction.')
                # precision ,recall ,f1 score.
                precision = Precision(prediction,labels,label)
                recall = Recall(prediction,labels,label)
                f1_score = F1_Score(precision,recall)
                TensorFlow_log.info('Precision: %.2f ,Recall: %.2f , F1: %.2f',precision,recall,f1_score)
                TensorFlow_log.info('Save in file.')
    TensorFlow_log.info('Finish all user predict and save.')

            
 

    # a = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    # for tensor_name in a:
    #     print(tensor_name,'\n')