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
from params import *

def vgg(N_LABELS,  batch_size = None, dtype='float32'):
	input_shape = (HEIGHT, WIDTH, CHANNEL)
	convdepth 	= [8, 16, 32, 64]
	fcdepth 	= [256, 128]
	BNaxis 		= 3	



	img_input = layers.Input( 	shape = input_shape, 
								dtype = dtype, 
								batch_size = batch_size)	


	x = layers.Conv2D(	convdepth[0], (3, 3),
						strides 	= (1, 1),
						padding 	= 'same', 
						use_bias 	= True,
						bias_initializer 	= 'zeros',
						kernel_initializer 	= 'he_normal',
						#kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
						name='conv1_1')(img_input) 


	x = layers.BatchNormalization(	axis 	 = BNaxis,
									momentum = BATCH_NORM_DECAY,
									epsilon  = BATCH_NORM_EPSILON,
									name 	 = 'bn1_1')(x)		

	x = layers.Activation('relu')(x)				
	
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)


	x = layers.Conv2D(	convdepth[1], (3, 3),
						strides 	= (1, 1),
						padding 	= 'same', 
						use_bias 	= True,
						bias_initializer 	= 'zeros',
						kernel_initializer 	= 'he_normal',
						#kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
						name='conv2_1')(x) 


	x = layers.BatchNormalization(	axis 	 = BNaxis,
									momentum = BATCH_NORM_DECAY,
									epsilon  = BATCH_NORM_EPSILON,
									name 	 = 'bn2_1')(x)		

	x = layers.Activation('relu')(x)				
	
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)


	x = layers.Conv2D(	convdepth[2], (3, 3),
						strides 	= (1, 1),
						padding 	= 'same', 
						use_bias 	= True,
						bias_initializer 	= 'zeros',
						kernel_initializer 	= 'he_normal',
						#kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
						name='conv3_1')(x) 


	x = layers.BatchNormalization(	axis 	 = BNaxis,
									momentum = BATCH_NORM_DECAY,
									epsilon  = BATCH_NORM_EPSILON,
									name 	 = 'bn3_1')(x)		

	x = layers.Activation('relu')(x)				
	
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)


	x = layers.Conv2D(	convdepth[3], (3, 3),
						strides 	= (1, 1),
						padding 	= 'same', 
						use_bias 	= True,
						bias_initializer 	= 'zeros',
						kernel_initializer 	= 'he_normal',
						#kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
						name='conv4_1')(x) 


	x = layers.BatchNormalization(	axis 	 = BNaxis,
									momentum = BATCH_NORM_DECAY,
									epsilon  = BATCH_NORM_EPSILON,
									name 	 = 'bn4_1')(x)		

	x = layers.Activation('relu')(x)				
	
	x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)


	x = layers.Flatten()(x)

	x = layers.Dense( 	fcdepth[0],  
						kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
						bias_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
						name='fc_1')(x)	

	x = layers.Dense( 	fcdepth[1],  
						kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
						bias_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
						name='fc_2')(x)

	x = layers.Dense( 	N_LABELS,  
						kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
						bias_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
						name='fc_3')(x)
	x = backend.cast(x, 'float32')

	x = layers.Activation('softmax')(x)

	return models.Model(img_input, x, name = 'vgg')




def cnn(N_LABELS,  batch_size = None, dtype='float32'):
    img_input = layers.Input( 	shape = input_shape, dtype = dtype, batch_size = batch_size)	

    x = layers.Conv2D(	128, 3,
                        strides 	= (1, 1),
                        padding 	= 'same',
                        activation 	= 'relu',
                        use_bias 	= True,
                        bias_initializer 	= 'zeros',
                        kernel_initializer 	= 'he_normal')(img_input) 

    x = layers.Conv2D(	128, 3,
                        strides 	= (1, 1),
                        padding 	= 'same',
                        activation 	= 'relu',
                        use_bias 	= True,
                        bias_initializer 	= 'zeros',
                        kernel_initializer 	= 'he_normal')(x) 

    x = layers.Conv2D(	128, 3,
                        strides 	= (1, 1),
                        padding 	= 'same',
                        activation 	= 'relu',
                        use_bias 	= True,
                        bias_initializer 	= 'zeros',
                        kernel_initializer 	= 'he_normal')(x) 

    x = layers.Conv2D(	128, 3,
                        strides 	= (1, 1),
                        padding 	= 'same',
                        activation 	= 'relu',
                        use_bias 	= True,
                        bias_initializer 	= 'zeros',
                        kernel_initializer 	= 'he_normal')(x) 

    x = layers.Conv2D(	128, 3,
                        strides 	= (1, 1),
                        padding 	= 'same',
                        activation 	= 'relu',
                        use_bias 	= True,
                        bias_initializer 	= 'zeros',
                        kernel_initializer 	= 'he_normal')(x) 

    x = layers.Conv2D(	128, 3,
                        strides 	= (1, 1),
                        padding 	= 'same',
                        activation 	= 'relu',
                        use_bias 	= True,
                        bias_initializer 	= 'zeros',
                        kernel_initializer 	= 'he_normal')(x) 

    x = layers.Conv2D(	128, 3,
                        strides 	= (1, 1),
                        padding 	= 'same',
                        activation 	= 'relu',
                        use_bias 	= True,
                        bias_initializer 	= 'zeros',
                        kernel_initializer 	= 'he_normal')(x) 

    x = layers.Conv2D(	128, 3,
                        strides 	= (1, 1),
                        padding 	= 'same',
                        activation 	= 'relu',
                        use_bias 	= True,
                        bias_initializer 	= 'zeros',
                        kernel_initializer 	= 'he_normal')(x) 

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = layers.Flatten()(x)

    x = layers.Dense( 	fcdepth[0],  
                        kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
                        bias_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY))(x)

    x = layers.Dense( 	fcdepth[1],  
                        kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
                        bias_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY))(x)

    x = layers.Dense( 	N_LABELS,  
                        kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
                        bias_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY))(x)

    x = backend.cast(x, 'float32')

    x = layers.Activation('softmax')(x)

    model = models.Model(img_input, x, name = 'without_skip')
    
    return model



def resnet(N_LABELS,  batch_size = None, dtype='float32'):
    img_input = layers.Input( 	shape = input_shape, dtype = dtype, batch_size = batch_size)	
    x = img_input

    _x = layers.Conv2D(	128, 3,
                        strides 	= (1, 1),
                        padding 	= 'same',
                        activation 	= 'relu',
                        use_bias 	= True,
                        bias_initializer 	= 'zeros',
                        kernel_initializer 	= 'he_normal')(x) 

    _x = layers.Conv2D(	128, 3,
                        strides 	= (1, 1),
                        padding 	= 'same',
                        activation 	= 'relu',
                        use_bias 	= True,
                        bias_initializer 	= 'zeros',
                        kernel_initializer 	= 'he_normal')(x) 
    x = x + _x

    _x = layers.Conv2D(	128, 3,
                        strides 	= (1, 1),
                        padding 	= 'same',
                        activation 	= 'relu',
                        use_bias 	= True,
                        bias_initializer 	= 'zeros',
                        kernel_initializer 	= 'he_normal')(x) 

    _x = layers.Conv2D(	128, 3,
                        strides 	= (1, 1),
                        padding 	= 'same',
                        activation 	= 'relu',
                        use_bias 	= True,
                        bias_initializer 	= 'zeros',
                        kernel_initializer 	= 'he_normal')(_x) 

    x = x + _x

    _x = layers.Conv2D(	128, 3,
                        strides 	= (1, 1),
                        padding 	= 'same',
                        activation 	= 'relu',
                        use_bias 	= True,
                        bias_initializer 	= 'zeros',
                        kernel_initializer 	= 'he_normal')(x) 

    _x = layers.Conv2D(	128, 3,
                        strides 	= (1, 1),
                        padding 	= 'same',
                        activation 	= 'relu',
                        use_bias 	= True,
                        bias_initializer 	= 'zeros',
                        kernel_initializer 	= 'he_normal')(_x) 

    x = x + _x

    _x = layers.Conv2D(	128, 3,
                        strides 	= (1, 1),
                        padding 	= 'same',
                        activation 	= 'relu',
                        use_bias 	= True,
                        bias_initializer 	= 'zeros',
                        kernel_initializer 	= 'he_normal')(x) 

    _x = layers.Conv2D(	128, 3,
                        strides 	= (1, 1),
                        padding 	= 'same',
                        activation 	= 'relu',
                        use_bias 	= True,
                        bias_initializer 	= 'zeros',
                        kernel_initializer 	= 'he_normal')(_x) 

    x = x + _x

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)

    x = layers.Flatten()(x)

    x = layers.Dense( 	fcdepth[0],  
                        kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
                        bias_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY))(x)

    x = layers.Dense( 	fcdepth[1],  
                        kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
                        bias_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY))(x)

    x = layers.Dense( 	N_LABELS,  
                        kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
                        bias_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY))(x)

    x = backend.cast(x, 'float32')

    x = layers.Activation('softmax')(x)

    model = models.Model(img_input, x, name = 'without_skip')
    
    return model