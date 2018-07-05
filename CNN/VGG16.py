import tensorflow as tf
import numpy as np
from cnn import Vgg16
from skimage import io
from TFRecord import get_File,TFRecord_Writer,TFRecord_Reader
from utilities.log import TensorFlow_log
from utilities.util import F1_Score,Precision,Recall

# global value.
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640
IMAGE_DEPTH = 3
one_hot_depth = 6

if __name__ =='__main__':
    vgg = Vgg16(vgg16_npy_path='./utilities/vgg16.npy',output_layer_units=6,LR=0.001)
    vgg.train()
