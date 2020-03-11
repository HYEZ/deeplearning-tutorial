from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import numpy as np 
import os 
from imutils import paths
import cv2



from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras import utils
from tensorflow.keras import optimizers

from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

L2_WEIGHT_DECAY 	= 1e-4
BATCH_NORM_DECAY 	= 0.9
BATCH_NORM_EPSILON = 1e-5

HEIGHT	 	= 28
WIDTH  		= 28
CHANNEL 	= 1
input_shape = (HEIGHT, WIDTH, CHANNEL)

OFFSET 		= 1

BATCH_SIZE  = 32


path 		= '/mnt/c/Users/Z440_user1/Desktop/dataset/mnistNet/data/trainingSet'
path_model  = '/mnt/c/Users/Z440_user1/Desktop/dataset/mnistNet/mnistNet_resnet.hdf5'
LABEL 	= np.array(['0','1','2','3','4','5','6','7','8','9'])

FIXED_BATCH_SIZE = True

convdepth 	= [8, 16, 32, 64]
fcdepth 	= [256, 128]
BNaxis 		= 3	