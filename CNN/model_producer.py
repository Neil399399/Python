from TFRecord import get_File,TFRecord_Writer,TFRecord_Reader
from utilities.log import TensorFlow_log
from cnn import CNN_Model
import tensorflow as tf
import numpy as np
# global value.
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
IMAGE_DEPTH = 3
one_hot_depth = 5
LR = 0.001


if __name__ =='__main__':
    # train data.
    train_images,train_labels = TFRecord_Reader('./TFRecord/train.tfrecord',IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,30)
    test_images,test_labels = TFRecord_Reader('./TFRecord/test.tfrecord0',IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,20)

    # setting placeholder.
    tf_x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH],name='tf_x')/255
    image = tf.reshape(tf_x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])              # (batch, height, width, channel)
    tf_y = tf.placeholder(tf.float32, [None,one_hot_depth]) 
    output = CNN_Model(image,IMAGE_HEIGHT,IMAGE_WIDTH,6,36,2,'same',tf.nn.relu,one_hot_depth)

    # def.
    loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)
    accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

    # turn on tensorflow.
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # open queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    sess.run(init_op)

    # set train dict.
    train_feature, train_label = sess.run([train_images,train_labels])
    # decode train_label to one_hot.
    train_label_onehot = sess.run(tf.one_hot(train_label,one_hot_depth))

    # set test dict.
    test_feature, test_label = sess.run([test_images,test_labels])
    # decode test_label to one_hot.
    test_label_onehot = sess.run(tf.one_hot(test_label,one_hot_depth))
    
    # saver.
    saver = tf.train.Saver(max_to_keep=1)

    # training.
    TensorFlow_log.info('Make graph and start trainng.')
    for step in range(200):
        TensorFlow_log.info('Training step :%d',step)
        _, loss_ = sess.run([train_op, loss], {tf_x: train_feature, tf_y: train_label_onehot})
        if step % 10 == 0:
            validate_accuracy = sess.run(accuracy,{tf_x: train_feature, tf_y: train_label_onehot})
            TensorFlow_log.info('After %d training step(s), the validation accuracy is %.2f.',step,validate_accuracy)
            TensorFlow_log.info('loss : %s',loss_)

    # final validate with used test data.
    TensorFlow_log.info('Start Testing.')
    test_accuracy = sess.run(accuracy,{tf_x: test_feature, tf_y: test_label_onehot})
    TensorFlow_log.info('Final accuracy : %.2f',test_accuracy)
    test_output = sess.run(output, {tf_x: test_feature[:10]})
    pred_y = np.argmax(test_output, 1)
    TensorFlow_log.info('Prediction label : %s',pred_y)
    TensorFlow_log.info('Real label : %s',test_label[:10])
    # stop all threads
    coord.request_stop()
    coord.join(threads)
    TensorFlow_log.info('CNN training done.')

    # save model.
    saver.save(sess,'./Model/CNN.model',global_step=step)
