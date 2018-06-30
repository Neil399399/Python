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
one_hot_depth = 10
LR = 0.05
dropout = 0.4


if __name__ =='__main__':
    # train data.
    train_images,train_labels = TFRecord_Reader('./TFRecord/train.tfrecord',IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,40)
    test_images,test_labels = TFRecord_Reader('./TFRecord/test.tfrecord',IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,40)


    # setting placeholder.
    with tf.name_scope('Input'):
        tf_x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH],name='tf_x')/255
        image = tf.reshape(tf_x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH])              # (batch, height, width, channel)
        tf_y = tf.placeholder(tf.float32, [None,one_hot_depth],name='tf_y')
        keep_prob = tf.placeholder(tf.float32,name='dropout')

    # CNN.
    result = CNN_Model(image,IMAGE_HEIGHT,IMAGE_WIDTH,6,36,108,216,2,'same',tf.nn.relu,one_hot_depth)
    output = tf.nn.dropout(result,keep_prob)

    # def loss, accuracy.
    with tf.name_scope('Loss'):
        loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
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
    # tensorboard.
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('TensorBoard/train/',graph=sess.graph)
    test_writer = tf.summary.FileWriter('TensorBoard/test/',graph=sess.graph)


    # training.
    TensorFlow_log.info('Make graph and start trainng.')
    for step in range(201):
        TensorFlow_log.info('Training step :%d',step)
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
    temp_accuracy = 0
    temp_precision = 0
    temp_recall = 0

    test_output = sess.run(output, {tf_x: test_feature,keep_prob:dropout})
    predictions = np.argmax(test_output, 1)
    labels = np.argmax(test_label_onehot, 1)
    test_summary_acc,test_accuracy = sess.run([merged,accuracy],{tf_x: test_feature, tf_y: test_label_onehot,keep_prob:dropout})
    test_precision = sess.run(precision,{tf_x: test_feature, tf_y: test_label_onehot,keep_prob:dropout})
    test_recall = sess.run(recall,{tf_x: test_feature, tf_y: test_label_onehot,keep_prob:dropout})
    test_writer.add_summary(test_summary_acc,step)

    print(test_precision,test_recall)
    # count each label precision and recall.
    # 0
    test_precision_label0 = Precision(predictions,labels,0)
    test_recall_label0 = Recall(predictions,labels,0)
    # 1
    test_precision_label1 = Precision(predictions,labels,1)
    test_recall_label1 = Recall(predictions,labels,1)
    # 2
    test_precision_label2 = Precision(predictions,labels,2)
    test_recall_label2 = Recall(predictions,labels,2)
    # 3
    test_precision_label3 = Precision(predictions,labels,3)
    test_recall_label3 = Recall(predictions,labels,3)
    # 4
    test_precision_label4 = Precision(predictions,labels,4)
    test_recall_label4 = Recall(predictions,labels,4)
    # 5
    test_precision_label5 = Precision(predictions,labels,5)
    test_recall_label5 = Recall(predictions,labels,5)
    # 6
    test_precision_label6 = Precision(predictions,labels,6)
    test_recall_label6 = Recall(predictions,labels,6)
    # 7
    test_precision_label7 = Precision(predictions,labels,7)
    test_recall_label7 = Recall(predictions,labels,7)
    # 8
    test_precision_label8 = Precision(predictions,labels,8)
    test_recall_label8 = Recall(predictions,labels,8)
    # 9
    test_precision_label9 = Precision(predictions,labels,9)
    test_recall_label9 = Recall(predictions,labels,9)

    # Final model info.
  
    TensorFlow_log.info('Test accuracy : %.2f',test_accuracy)
    print('--------------------------------------------------------------')
    TensorFlow_log.info('Class 0 precision : %.2f',test_precision_label0)
    TensorFlow_log.info('Class 0 recall : %.2f',test_recall_label0)
    TensorFlow_log.info('Class 0 f1 score : %.2f',F1_Score(test_precision_label0,test_recall_label0))

    print('--------------------------------------------------------------')
    TensorFlow_log.info('Class 1 precision : %.2f',test_precision_label1)
    TensorFlow_log.info('Class 1 recall : %.2f',test_recall_label1)
    TensorFlow_log.info('Class 1 f1 score : %.2f',F1_Score(test_precision_label1,test_recall_label1))

    print('--------------------------------------------------------------')
    TensorFlow_log.info('Class 2 precision : %.2f',test_precision_label2)
    TensorFlow_log.info('Class 2 recall : %.2f',test_recall_label2)
    TensorFlow_log.info('Class 2 f1 score : %.2f',F1_Score(test_precision_label2,test_recall_label2))

    print('--------------------------------------------------------------')
    TensorFlow_log.info('Class 3 precision : %.2f',test_precision_label3)
    TensorFlow_log.info('Class 3 recall : %.2f',test_recall_label3)
    TensorFlow_log.info('Class 3 f1 score : %.2f',F1_Score(test_precision_label3,test_recall_label3))

    print('--------------------------------------------------------------')
    TensorFlow_log.info('Class 4 precision : %.2f',test_precision_label4)
    TensorFlow_log.info('Class 4 recall : %.2f',test_recall_label4)
    TensorFlow_log.info('Class 4 f1 score : %.2f',F1_Score(test_precision_label4,test_recall_label4))
    
    print('--------------------------------------------------------------')
    TensorFlow_log.info('Class 5 precision : %.2f',test_precision_label5)
    TensorFlow_log.info('Class 5 recall : %.2f',test_recall_label5)
    TensorFlow_log.info('Class 5 f1 score : %.2f',F1_Score(test_precision_label4,test_recall_label5))

    print('--------------------------------------------------------------')
    TensorFlow_log.info('Class 6 precision : %.2f',test_precision_label6)
    TensorFlow_log.info('Class 6 recall : %.2f',test_recall_label6)
    TensorFlow_log.info('Class 6 f1 score : %.2f',F1_Score(test_precision_label4,test_recall_label6))

    print('--------------------------------------------------------------')
    TensorFlow_log.info('Class 7 precision : %.2f',test_precision_label7)
    TensorFlow_log.info('Class 7 recall : %.2f',test_recall_label7)
    TensorFlow_log.info('Class 7 f1 score : %.2f',F1_Score(test_precision_label4,test_recall_label7))

    print('--------------------------------------------------------------')
    TensorFlow_log.info('Class 8 precision : %.2f',test_precision_label8)
    TensorFlow_log.info('Class 8 recall : %.2f',test_recall_label8)
    TensorFlow_log.info('Class 8 f1 score : %.2f',F1_Score(test_precision_label4,test_recall_label8))

    print('--------------------------------------------------------------')
    TensorFlow_log.info('Class 9 precision : %.2f',test_precision_label9)
    TensorFlow_log.info('Class 9 recall : %.2f',test_recall_label9)
    TensorFlow_log.info('Class 9 f1 score : %.2f',F1_Score(test_precision_label4,test_recall_label9))
    # close queue.
    coord.request_stop()
    coord.join(threads)
    TensorFlow_log.info('CNN training done.')

    # save model.
    # saver.save(sess,'./Model/CNN.model',global_step=step)
