from TFRecord import TFRecord_Reader
from skimage import io
from utilities.util import Writer, Precision, Recall, F1_Score, Accuracy
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
label = 1

if __name__ =='__main__':
    
    test_images,test_labels = TFRecord_Reader('./TFRecord/test.tfrecord',IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,80)

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
    # for dirPath_root, dirNames, fileNames in os.walk(file_dir):
    #     for name in dirNames:
    #         subfolder_path = os.path.join(dirPath_root, name)
    sess = tf.Session()
    saver.restore(sess,tf.train.latest_checkpoint('./Model/'))
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # open queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    sess.run(init_op)

    # start predict.
    temp_predict = []
    temp_label = []
    for step in range(5):
        # set test dict.
        test_feature, test_label = sess.run([test_images,test_labels])
        test_output = sess.run(output, {tf_x: test_feature})
        prediction = np.argmax(test_output, 1)
        TensorFlow_log.info('Finish prediction.')
        for value in prediction:
            temp_predict.append(value)
        for label in test_label:
            temp_label.append(int(label))
    # precision ,recall ,f1 score.
    precision = Precision(temp_predict,temp_label,label)
    recall = Recall(prediction,test_label,label)
    f1_score = F1_Score(precision,recall)
    accuracy = Accuracy(prediction,test_label)
    TensorFlow_log.info('Precision: %.2f ,Recall: %.2f , F1: %.2f ,Acc: %.2f',precision,recall,f1_score,accuracy)

    coord.request_stop()
    coord.join(threads)
    TensorFlow_log.info('Save in file.')
    TensorFlow_log.info('Finish all user predict and save.')

            
 

    # a = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
    # for tensor_name in a:
    #     print(tensor_name,'\n')