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

from util import *
from model import *
from params import *






def resNet(N_LABELS,  batch_size = None, dtype='float32'):
	input_shape = (HEIGHT, WIDTH, CHANNEL)
	convdepth 	= [8, 16, 32, 64]
	fcdepth 	= [256, 128]
	BNaxis 		= 3	

	img_input = layers.Input( 	shape = input_shape, 
								dtype = dtype, 
								batch_size = batch_size)



	





if __name__ == '__main__':
	train()
	# inference()