import tensorflow as tf

def CNN_Model(Image, Image_height, Image_width, Conv1_Filter, Conv2_Filter, Pool_Size, Padding, Activation_Function, output_layer_units):
    # CNN.
    output_shape = (Image_height/Pool_Size**2)*(Image_width/Pool_Size**2)*Conv2_Filter
    # (image_height, image_width, Conv1_Filter)
    conv1 = tf.layers.conv2d(inputs=Image, filters=Conv1_Filter, kernel_size=5, strides=1, padding=Padding, activation=Activation_Function)
    pool1 = tf.layers.max_pooling2d(conv1,pool_size=Pool_Size,strides=2)  # -> (image_height/Pool_Size, image_width/Pool_Size, Conv1_Filter)
    # (image_height, image_width, Conv2_Filter)
    conv2 = tf.layers.conv2d(pool1, Conv2_Filter, 5, 1, Padding, activation=Activation_Function)    
    pool2 = tf.layers.max_pooling2d(conv2, Pool_Size, 2)

    flat = tf.reshape(pool2, [-1, int(output_shape)])
    # output layer
    output = tf.layers.dense(flat, output_layer_units,name='output')
    return output