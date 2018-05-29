from TFRecord import get_File,TFRecord_Writer,TFRecord_Reader
from cnn import CNN_Model
import tensorflow as tf
import numpy as np
# global value.
image_Dir = './example_data/'
image_folder_list = ['台灣','美食','捷運','早餐']
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
IMAGE_DEPTH = 3
one_hot_depth = 2
LR = 0.001


if __name__ =='__main__':
    # # make image to .TFRecord file.
    image_list,label_list = get_File(image_Dir)
    TFRecord_Writer(image_list,label_list,image_Dir,image_folder_list,'test.tfrecords')

    # train data.
    train_images,train_labels = TFRecord_Reader('test.tfrecords',IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_DEPTH,100)

    # setting placeholder.
    tf_x = tf.placeholder(tf.float32, [None, 640,640,3])/255
    image = tf.reshape(tf_x, [-1, 640, 640, 3])              # (batch, height, width, channel)
    tf_y = tf.placeholder(tf.float32, [None,one_hot_depth]) 
    output = CNN_Model(image,640,640,6,36,2,'same',tf.nn.relu,one_hot_depth)

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
    train_feature, train_label = sess.run([train_images[:1000],train_labels[:1000]])
    # decode train_label to one_hot.
    train_label_onehot = sess.run(tf.one_hot(train_label,one_hot_depth))

    # set test dict.
    test_feature, test_label = sess.run([train_images[1000:],train_labels[1000:]])
    # decode test_label to one_hot.
    test_label_onehot = sess.run(tf.one_hot(train_label,one_hot_depth))
    
    # training.
    print('Start training ... ')
    for step in range(500):
        _, loss_ = sess.run([train_op, loss], {tf_x: train_feature, tf_y: train_label_onehot})
        if step % 50 == 0:
            validate_accuracy = sess.run(accuracy, {tf_x: train_feature, tf_y: train_label_onehot})
            print('After %d training step(s), the validation accuracy is %.2f.'%(step,validate_accuracy))
            print('loss : ',loss_)

    # final validate with used test data.
    print('Start Testing ... ')
    test_accuracy = sess.run(accuracy,{tf_x: test_feature, tf_y: test_label_onehot})
    print('final accuracy : ',test_accuracy)
    test_output = sess.run(output, {tf_x: test_feature[:2]})
    pred_y = np.argmax(test_output, 1)
    print('prediction label',pred_y)
    print('real label',test_label[:2])

    # stop all threads
    coord.request_stop()
    coord.join(threads)