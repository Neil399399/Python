import tensorflow as tf
import numpy as np
from TFRecord import get_File,TFRecord_Writer,TFRecord_Reader
from utilities.log import TensorFlow_log

def CNN_Model(Image, Image_height, Image_width, Conv1_Filter, Conv2_Filter, Conv3_Filter ,Conv4_Filter,Pool_Size, Padding, Activation_Function, output_layer_units):
    # CNN.
    output_shape = (Image_height/Pool_Size**4)*(Image_width/Pool_Size**4)*Conv4_Filter
    # (image_height, image_width, Conv1_Filter)
    conv1 = tf.layers.conv2d(inputs=Image, filters=Conv1_Filter, kernel_size=5, strides=1, padding=Padding, activation=Activation_Function,name='conv1')
    pool1 = tf.layers.max_pooling2d(conv1,pool_size=Pool_Size,strides=2,name='pool1')  # -> (image_height/Pool_Size, image_width/Pool_Size, Conv1_Filter)
    # (image_height, image_width, Conv2_Filter)
    conv2 = tf.layers.conv2d(pool1, Conv2_Filter, 3, 1, Padding, activation=Activation_Function,name='conv2')    
    pool2 = tf.layers.max_pooling2d(conv2, Pool_Size, 2,name='pool2')

    conv3 = tf.layers.conv2d(pool2, Conv3_Filter, 3, 1, Padding, activation=Activation_Function,name='conv3')    
    pool3 = tf.layers.max_pooling2d(conv3, Pool_Size, 2,name='pool3')

    conv4 = tf.layers.conv2d(pool3, Conv4_Filter, 3, 1, Padding, activation=Activation_Function,name='conv4')    
    pool4 = tf.layers.max_pooling2d(conv4, Pool_Size, 2,name='pool4')

    flat = tf.reshape(pool4, [-1, int(output_shape)])
    # output layer
    fc5 = tf.layers.dense(flat, 256,tf.nn.relu ,name='fc5')
    output = tf.layers.dense(fc5, output_layer_units, name='output')
    
    return output

# VGG 16
class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None, restore_from=None,output_layer_units=None,LR=None):

        TensorFlow_log.info('load train data')
        # load data.
        self.train_images,self.train_labels = TFRecord_Reader('./TFRecord/train.tfrecord',640,640,3,80)
        self.test_images,self.test_labels = TFRecord_Reader('./TFRecord/test.tfrecord',640,640,3,30)
        
        # pre-trained parameters
        TensorFlow_log.info('Start pretrain')
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('Please download VGG16 parameters from here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM\nOr from my Baidu Cloud: https://pan.baidu.com/s/1Spps1Wy0bvrQHH2IMkRfpg')
        with tf.name_scope('Input'):
            self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
            self.tfy = tf.placeholder(tf.float32, [None, output_layer_units])

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])

        # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')

        # detach original VGG fc layers and
        # reconstruct your own fc layers serve for your own purpose
        self.flatten = tf.reshape(pool5, [-1, 7*7*512])
        self.fc6 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name='fc6')
        self.out = tf.layers.dense(self.fc6, output_layer_units, name='output')

        self.sess = tf.Session()
        # open queue.
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(sess=self.sess,coord=self.coord)


        if restore_from:
            saver = tf.train.Saver()
            saver.restore(self.sess, restore_from)
        else:   # training graph
            with tf.name_scope('Loss'):
                self.loss = tf.losses.mean_squared_error(labels=self.tfy, predictions=self.out)
            tf.summary.scalar('loss',self.loss)
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.loss)
            with tf.name_scope('Accuracy'):
                self.accuracy = tf.metrics.accuracy(labels=tf.argmax(self.tfy, axis=1), predictions=tf.argmax(self.out, axis=1))[1]
            tf.summary.scalar('accuracy',self.accuracy)
            with tf.name_scope('Precision'):
                self.precision = tf.metrics.precision(labels=tf.argmax(self.tfy, axis=1), predictions=tf.argmax(self.out, axis=1))[1]
            tf.summary.scalar('precision',self.precision)
            with tf.name_scope('Recall'):
                self.recall = tf.metrics.recall(labels=tf.argmax(self.tfy, axis=1), predictions=tf.argmax(self.out, axis=1))[1]
            tf.summary.scalar('recall',self.recall)

            # tensorboard.
            self.merged = tf.summary.merge_all()
            self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.sess.run(self.init_op)
            self.train_writer = tf.summary.FileWriter('TensorBoard/train/',graph=self.sess.graph)
            self.test_writer = tf.summary.FileWriter('TensorBoard/test/',graph=self.sess.graph)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def train(self,learning_step,output_layer_units):
        for step in range(learning_step): 
            # set train dict.
            self.train_feature, self.train_label = self.sess.run([self.train_images,self.train_labels])
            # decode train_label to one_hot.
            self.train_label_onehot = self.sess.run(tf.one_hot(self.train_label,output_layer_units))
            # set test dict.
            self.test_feature, self.test_label = self.sess.run([self.test_images,self.test_labels])
            # decode test_label to one_hot.
            self.test_label_onehot = self.sess.run(tf.one_hot(self.test_label,output_layer_units))
            
            loss, _ = self.sess.run([self.loss, self.train_op], {self.tfx: self.train_feature, self.tfy: self.train_label_onehot})
            summary_tloss, _ = self.sess.run([self.merged, self.loss], {self.tfx: self.train_feature, self.tfy: self.train_label_onehot})
            TensorFlow_log.info('Step %d , Loss: %s',step,loss)
            self.train_writer.add_summary(summary_tloss,step)

            if step %10 == 0:
                loss, accuracy, precision, recall, summary_loss, summary_acc, summary_pre,summary_rec = self.validate()
                self.test_writer.add_summary(summary_loss,step)
                self.test_writer.add_summary(summary_acc,step)
                self.test_writer.add_summary(summary_pre,step)
                self.test_writer.add_summary(summary_rec,step)
                TensorFlow_log.info('Loss: %s ,Acc: %.2f ,Precision: %.2f ,Recall: %.2f ',loss, accuracy, precision, recall)
        # close queue.
        self.coord.request_stop()
        self.coord.join(self.threads)
        TensorFlow_log.info('CNN training done.')
                

    def validate(self):
        summary_loss, loss = self.sess.run([self.merged, self.loss], {self.tfx: self.test_feature, self.tfy: self.test_label_onehot})
        summary_acc, accuracy = self.sess.run([self.merged, self.accuracy],{self.tfx: self.test_feature, self.tfy: self.test_label_onehot})
        summary_pre, precision = self.sess.run([self.merged, self.precision],{self.tfx: self.test_feature, self.tfy: self.test_label_onehot})
        summary_rec,recall = self.sess.run([self.merged, self.recall],{self.tfx: self.test_feature, self.tfy: self.test_label_onehot})
        return loss, accuracy, precision, recall, summary_loss, summary_acc, summary_pre,summary_rec

    # def predict(self, paths):
    #     fig, axs = plt.subplots(1, 2)
    #     for i, path in enumerate(paths):
    #         x = load_img(path)
    #         length = self.sess.run(self.out, {self.tfx: x})
    #         axs[i].imshow(x[0])
    #         axs[i].set_title('Len: %.1f cm' % length)
    #         axs[i].set_xticks(()); axs[i].set_yticks(())
    #     plt.show()

    def save(self, path='./for_transfer_learning/model/transfer_learn'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)
