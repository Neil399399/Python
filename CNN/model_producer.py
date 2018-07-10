from TFRecord import get_File,TFRecord_Writer,TFRecord_Reader
from utilities.log import TensorFlow_log
from utilities.util import F1_Score,Precision,Recall
from cnn import CNN_Model
import tensorflow as tf
import numpy as np
# global value.
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
IMAGE_DEPTH = 3
one_hot_depth = 6
LR = 1e-4
dropout = 0.4


if __name__ =='__main__':
    # train data.
    train_images,train_labels = TFRecord_Reader('./TFRecord/train.tfrecord',IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,100)

    # setting placeholder.
    with tf.name_scope('Input'):
        tf_x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH],name='tf_x')
        image = tf.reshape(tf_x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])              # (batch, height, width, channel)
        tf_y = tf.placeholder(tf.float32, [None,one_hot_depth],name='tf_y')
        keep_prob = tf.placeholder(tf.float32,name='dropout')

    # CNN.
    result = CNN_Model(image,IMAGE_HEIGHT,IMAGE_WIDTH,3,26,72,128,2,'same',tf.nn.relu,one_hot_depth)
    output = tf.nn.dropout(result,keep_prob)

    # def loss, accuracy.
    with tf.name_scope('Loss'):
        # loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
        loss = tf.reduce_mean(-tf.reduce_sum(tf_y * tf.log((output +1)/2+ 1e-15),
                                                  reduction_indices=[1]))
    tf.summary.scalar('loss',loss)

    with tf.name_scope('Train'):
        train_op = tf.train.GradientDescentOptimizer(LR).minimize(loss)
    tf.summary.scalar('leaning_rate',LR)

    with tf.name_scope('Accuracy'):
        accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),name='accuracy')[1]
    tf.summary.scalar('accuracy',accuracy)

    with tf.name_scope('Precision'):
        precision = tf.metrics.precision(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),name='precision')[1]
    tf.summary.scalar('precision',precision)

    with tf.name_scope('Recall'):
        recall = tf.metrics.recall(labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),name='recall')[1]
    tf.summary.scalar('recall',recall)

    # turn on tensorflow.
    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # open queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    sess.run(init_op)

    # saver.
    saver = tf.train.Saver(max_to_keep=1)
    # tensorboard.
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('TensorBoard/train/',graph=sess.graph)
    test_writer = tf.summary.FileWriter('TensorBoard/test/',graph=sess.graph)


    # training.
    TensorFlow_log.info('Make graph and start training.')
    for step in range(201):
        TensorFlow_log.info('Training step :%d',step)
        # set train dict.
        train_feature, train_label = sess.run([train_images,train_labels])
        # decode train_label to one_hot.
        train_label_onehot = sess.run(tf.one_hot(train_label,one_hot_depth))

        _, loss_ = sess.run([train_op, loss], {tf_x: train_feature, tf_y: train_label_onehot,keep_prob:dropout})
        summary_loss,_ = sess.run([merged,loss],{tf_x: train_feature, tf_y: train_label_onehot,keep_prob:dropout})

        if step % 10 == 0:
            summary_acc,validate_accuracy = sess.run([merged,accuracy],{tf_x: train_feature, tf_y: train_label_onehot,keep_prob:dropout})
            TensorFlow_log.info('After %d training step(s), the validation accuracy is %.2f.',step,validate_accuracy)
            TensorFlow_log.info('loss : %s',loss_)
            train_writer.add_summary(summary_acc,step)
            train_writer.add_summary(summary_loss,step)

    # final validate with used test data.
    TensorFlow_log.info('Start Testing(100).')
    test_output = sess.run(output, {tf_x: train_feature,keep_prob:dropout})
    predictions = np.argmax(test_output, 1)
    labels = np.argmax(train_label_onehot, 1)
    test_summary_acc,test_accuracy = sess.run([merged,accuracy],{tf_x: train_feature, tf_y: train_label_onehot,keep_prob:dropout})
    test_precision = sess.run(precision,{tf_x: train_feature, tf_y: train_label_onehot,keep_prob:dropout})
    test_recall = sess.run(recall,{tf_x: train_feature, tf_y: train_label_onehot,keep_prob:dropout})
    test_writer.add_summary(test_summary_acc,step)
    print(test_precision,test_recall)
    # close queue.
    coord.request_stop()
    coord.join(threads)
    TensorFlow_log.info('CNN training done.')

    # save model.
    saver.save(sess,'./Model/CNN(6).model',global_step=step)
