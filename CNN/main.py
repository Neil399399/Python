from TFRecord import get_File,TFRecord_Writer,TFRecord_Reader
from cnn import CNN_Model
import tensorflow as tf
import numpy as np
# global value.
image_Dir = './example_data/'
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
IMAGE_DEPTH = 3


LR = 0.001



if __name__ =='__main__':
    # # make image to .TFRecord file.
    # images,labels = get_File(image_Dir)
    # TFRecord_Writer(images,labels,'./example_data/temp1/','test.tfrecords')
    images,labels = TFRecord_Reader('test.tfrecords',IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,1)

    tf_x = tf.placeholder(tf.float32, [1, 640,640,3])
    image = tf.reshape(tf_x, [-1, 640, 640, 3])              # (batch, height, width, channel)
    tf_y = tf.placeholder(tf.float32, [None, 1]) 
    flat, output = CNN_Model(image,640,640,6,36,2,'same',tf.nn.relu)

    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)
    accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    sess.run(init_op)

    for step in range(2):
        print(images.shape)
        w,y = sess.run([images,labels])
        print(w,y)
    #     _, loss_ = sess.run([train_op, loss], {tf_x: images, tf_y: labels})
    #     if step % 50 == 0:
    #         accuracy_, flat_representation = sess.run([accuracy, flat], {tf_x: images, tf_y: labels})

    # test_output = sess.run(output, {tf_x: images[:2]})
    # pred_y = np.argmax(test_output, 1)
    # print(pred_y, 'prediction number')
    # print(np.argmax(labels[:2], 1), 'real number')

    # stop all threads.
    coord.request_stop()
    coord.join(threads)