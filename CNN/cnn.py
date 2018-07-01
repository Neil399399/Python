import tensorflow as tf

def CNN_Model(Image, Image_height, Image_width, Conv1_Filter, Conv2_Filter, Conv3_Filter ,Pool_Size, Padding, Activation_Function, output_layer_units):
    # CNN.
    output_shape = (Image_height/Pool_Size**3)*(Image_width/Pool_Size**3)*Conv3_Filter
    # (image_height, image_width, Conv1_Filter)
    conv1 = tf.layers.conv2d(inputs=Image, filters=Conv1_Filter, kernel_size=5, strides=1, padding=Padding, activation=Activation_Function,name='conv1')
    pool1 = tf.layers.max_pooling2d(conv1,pool_size=Pool_Size,strides=2,name='pool1')  # -> (image_height/Pool_Size, image_width/Pool_Size, Conv1_Filter)
    # (image_height, image_width, Conv2_Filter)
    conv2 = tf.layers.conv2d(pool1, Conv2_Filter, 5, 1, Padding, activation=Activation_Function,name='conv2')    
    pool2 = tf.layers.max_pooling2d(conv2, Pool_Size, 2,name='pool2')

    conv3 = tf.layers.conv2d(pool2, Conv3_Filter, 5, 1, Padding, activation=Activation_Function,name='conv3')    
    pool3 = tf.layers.max_pooling2d(conv3, Pool_Size, 2,name='pool3')

    # conv4 = tf.layers.conv2d(pool3, Conv4_Filter, 5, 1, Padding, activation=Activation_Function,name='conv4')    
    # pool4 = tf.layers.max_pooling2d(conv4, Pool_Size, 2,name='pool4')

    flat = tf.reshape(pool3, [-1, int(output_shape)])
    # output layer
    output = tf.layers.dense(flat, output_layer_units,activation=tf.nn.softmax ,name='output')
    
    return output