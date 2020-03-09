import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Lambda, Layer, BatchNormalization, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np 
import os 
from imutils import paths
import cv2

class MaxUnpoolWithArgmax(Layer):
	def __init__(self, pooling_argmax, stride = [1, 2, 2, 1], **kwargs):
		self.pooling_argmax = pooling_argmax    
		self.stride = stride
		super(MaxUnpoolWithArgmax, self).__init__(**kwargs)

	def build(self, input_shape):
		super(MaxUnpoolWithArgmax, self).build(input_shape)


	def call(self, inputs):
		argmax 			= self.pooling_argmax 
		input_shape 	= K.cast(K.shape(inputs), dtype='int64') # Input Layer to be Unpooled

		output_shape 	= (	input_shape[0], 	input_shape[1] * self.stride[1],
							input_shape[2] * self.stride[2],	input_shape[3])  # To-be-Unpooled Layer

		flat_input_size 	= tf.cumprod(input_shape)[-1]
		#flat_input_size 	= input_shape[0] * input_shape[1]  * input_shape[2] * input_shape[3]
		flat_output_shape   = tf.stack([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])

		pool_ 				= tf.reshape(inputs, tf.stack([flat_input_size]))
		batch_range 		= tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype = argmax.dtype),
										shape = tf.stack([input_shape[0], 1, 1, 1]))

		b 					= tf.ones_like(argmax) * batch_range
		b 					= tf.reshape(b, tf.stack([flat_input_size, 1]))

		
		ind_ 				= tf.reshape(argmax, tf.stack([flat_input_size, 1]))
		ind_ 				= ind_ - b * tf.cast(flat_output_shape[1], tf.int64)
		ind_ 				= tf.concat([b, ind_], 1)

		ret 				= tf.scatter_nd(ind_, pool_, shape = tf.cast(flat_output_shape, tf.int64))
		ret 				= tf.reshape(ret, tf.stack(output_shape))

		set_input_shape     = inputs.get_shape()
		set_output_shape    = [	set_input_shape[0], set_input_shape[1] * self.stride[1], 
								set_input_shape[2] * self.stride[2], set_input_shape[3] ]
		ret.set_shape(set_output_shape)
		return ret


	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[1] * 2, input_shape[2] * 2, input_shape[3])

	def get_config(self):
		base_config 	= super(MaxUnpoolWithArgmax, self).get_config()
		base_config['pooling_argmax'] 	= self.pooling_argmax
		base_config['stride'] 			= self.stride
		return base_config

	@classmethod
	def from_config(cls, config):
		return cls(**config)


class DeconvNet:
	def __init__(self):
		self.input_shape 	= (96, 96, 1)
		self.nlabels 		= 5 + 1 # [0: bg, 1: face, 2: right eye, 3: left eye, 4: nose, 5: mouse]

	def predict(self, image):
		return self.model.predict(image)

	def save(self, file_path='model.h5'):
		#print(self.model.to_json())
		#self.model.save_weights(file_path)		
		self.model.save_weights(file_path)

	def load(self, file_path='model.h5'):
		self.model.load_weights(file_path)

	def max_pool_with_argmax(self, x):
		return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	def buildConv2DBlock(self, block_input, num_filters, block_number, num_blocks):
		for i in range(1, num_blocks + 1):
			if i == 1:
				conv2d = Conv2D(num_filters, 3, padding='same', name='conv{}-{}'.format(block_number, i), use_bias=False)(block_input)
			else:
				conv2d = Conv2D(num_filters, 3, padding='same', name='conv{}-{}'.format(block_number, i), use_bias=False)(conv2d)
			conv2d = BatchNormalization(name='batchnorm{}-{}'.format(block_number, i))(conv2d)
			conv2d = Activation('relu', name='relu{}-{}'.format(block_number, i))(conv2d)
		return conv2d


	def build_model(self, use_cpu=True, print_summary=True,  batch_size = None):
		print('building network')

		DEPTH = [8, 16, 32, 64, 256]

		if use_cpu:
			device = '/cpu:0'
		else:
			device = '/gpu:0'

		with tf.device(device):			
			if batch_size is not None:
				self.BATCH_SIZE = batch_size
				inputs = Input(shape = self.input_shape)#, batch_size = self.BATCH_SIZE)
			else:
				self.BATCH_SIZE = None
				inputs = Input(shape = self.input_shape)

			conv_block_1 = self.buildConv2DBlock(inputs, DEPTH[0], 1, 2) 						# [?, 96, 96, DEPTH0] x 2
			pool1, pool1_argmax = Lambda(self.max_pool_with_argmax, name='pool1')(conv_block_1) # [?, 48, 48, DEPTH0]

			conv_block_2 = self.buildConv2DBlock(pool1, DEPTH[1], 2, 2)  						# [?, 48, 48, DEPTH1] x 2
			pool2, pool2_argmax = Lambda(self.max_pool_with_argmax, name='pool2')(conv_block_2) # [?, 24, 24, DEPTH1]

			conv_block_3 = self.buildConv2DBlock(pool2, DEPTH[2], 3, 2)  						# [?, 24, 24, DEPTH2] x 2
			pool3, pool3_argmax = Lambda(self.max_pool_with_argmax, name='pool3')(conv_block_3) # [?, 12, 12, DEPTH2]

			conv_block_4 = self.buildConv2DBlock(pool3, DEPTH[3], 4, 2)  						# [?, 12, 12, DEPTH3] x 2
			pool4, pool4_argmax = Lambda(self.max_pool_with_argmax, name='pool4')(conv_block_4) # [?,  6,  6, DEPTH3]

			fc1 = Conv2D(DEPTH[-1], (6, 6), use_bias=False, padding='valid', name='fc1')(pool4) 	#[?, 1, 1, DEPTH4]
			fc1 = BatchNormalization(name='batchnorm_fc1')(fc1) 									
			fc1 = Activation('relu', name='relu_fc1')(fc1)											
			
			fc2 = Conv2D(DEPTH[-1], 1, use_bias=False, padding='valid', name='fc2')(fc1)   			#[?, 1, 1, DEPTH4]
			fc2 = BatchNormalization(name='batchnorm_fc2')(fc2) 									
			fc2 = Activation('relu', name='relu_fc2')(fc2)											

			x = Conv2DTranspose(DEPTH[3], (6, 6), use_bias=False, padding='valid', name='fc2-deconv')(fc2) 	#[?, 6, 6, DEPTH3]
			x = BatchNormalization(name='batchnorm_fc2-deconv')(x) 											
			x = Activation('relu', name='relu_fc2-deconv')(x)            										

			x = MaxUnpoolWithArgmax(pool4_argmax, name='unpool4')(x) 	 							#[?, 12, 12, DEPTH3]
			x = Conv2DTranspose(DEPTH[3], 3, use_bias=False, padding='same', name='deconv4-1')(x) 	#[?, 12, 12, DEPTH3]
			x = BatchNormalization(name='batchnorm_deconv4-1')(x) 									
			x = Activation('relu', name='relu_deconv4-1')(x)  										
			x = Conv2DTranspose(DEPTH[2], 3, use_bias=False, padding='same', name='deconv4-2')(x) 	#[?, 12, 12, DEPTH2]
			x = BatchNormalization(name='batchnorm_deconv4-2')(x) 									
			x = Activation('relu', name='relu_deconv4-2')(x)  		

			x = MaxUnpoolWithArgmax(pool3_argmax, name='unpool3')(x) 								#[?, 24, 24, DEPTH2]
			x = Conv2DTranspose(DEPTH[2], 3, use_bias=False, padding='same', name='deconv3-1')(x) 	#[?, 24, 24, DEPTH2]
			x = BatchNormalization(name='batchnorm_deconv3-1')(x) 									
			x = Activation('relu', name='relu_deconv3-1')(x)  										
			x = Conv2DTranspose(DEPTH[1], 3, use_bias=False, padding='same', name='deconv3-2')(x) 	#[?, 24, 24, DEPTH1]
			x = BatchNormalization(name='batchnorm_deconv3-2')(x) 									
			x = Activation('relu', name='relu_deconv3-2')(x)  

			x = MaxUnpoolWithArgmax(pool2_argmax, name='unpool2')(x) 								#[?, 48, 48, DEPTH1]
			x = Conv2DTranspose(DEPTH[1], 3, use_bias=False, padding='same', name='deconv2-1')(x) 	#[?, 48, 48, DEPTH1]
			x = BatchNormalization(name='batchnorm_deconv2-1')(x) 									
			x = Activation('relu', name='relu_deconv2-1')(x)  										
			x = Conv2DTranspose(DEPTH[0], 3, use_bias=False, padding='same', name='deconv2-2')(x) 	#[?, 48, 48, DEPTH0]
			x = BatchNormalization(name='batchnorm_deconv2-2')(x) 									
			x = Activation('relu', name='relu_deconv2-2')(x)   										

			x = MaxUnpoolWithArgmax(pool1_argmax, name='unpool1')(x) 								#[?, 96, 96, DEPTH0]
			x = Conv2DTranspose(DEPTH[0], 3, use_bias=False, padding='same', name='deconv1-1')(x) 	#[?, 96, 96, DEPTH0]
			x = BatchNormalization(name='batchnorm_deconv1-1')(x) 									
			x = Activation('relu', name='relu_deconv1-1')(x)   										
			x = Conv2DTranspose(DEPTH[0], 3, use_bias=False, padding='same', name='deconv1-2')(x) 	#[?, 96, 96, DEPTH0]
			x = BatchNormalization(name='batchnorm_deconv1-2')(x) 									
			x = Activation('relu', name='relu_deconv1-2')(x)   						

			output = Conv2DTranspose(self.nlabels, 1, activation='softmax', padding='same', name='output')(x) #[?, 96, 96, NLABELS]

			self.model = Model(inputs=inputs, outputs=output)

			if print_summary:
				print(self.model.summary())

			self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse'])	

			return self.model

		
	def read_train_data(self, path_train, trainfolder = '', segfolder = '', 
								train_sample_ratio = None, dtype = 'float', 
								ext_train = '.png', ext_seg = '.png', 
								img_regularization = True):
	
		imagePaths 		= list(paths.list_images(path_train + trainfolder))

		X 			= np.empty((len(imagePaths), self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype = dtype)
		Y 			= np.empty((len(imagePaths), self.input_shape[0], self.input_shape[1], self.nlabels), dtype = 'int')

		for (i, path) in enumerate(imagePaths):
			
			img = cv2.imread(path)
			
			if self.input_shape[2] == 1:
				im 		= np.array(img[:,:,0], dtype = dtype) 	
				im 		= np.expand_dims(im, axis = 2) 			
			else:
				im 		= np.array(image, dtype = dtype) 	

			if img_regularization:
				X[i] 	= im / 255.

			path_split 		= path.split(trainfolder)[0] + segfolder + path.split(trainfolder)[1]
			pathseg 		= path_split.split(ext_train)[0]  + ext_seg

			img 	= cv2.imread(pathseg)
			im 		= np.array(img[:,:,0], dtype = 'int') 	
			Y[i] 	= (np.arange(self.nlabels) == im[..., None]).astype(int)

		if train_sample_ratio is not None:
			Ntot 		= len(imagePaths)
			trX 		= X[:int(Ntot * train_sample_ratio),...]
			trY 		= Y[:int(Ntot * train_sample_ratio),...]
			teX 		= X[ int(Ntot * train_sample_ratio):,...]
			teY 		= Y[ int(Ntot * train_sample_ratio):,...]

			self.trX 	= trX[:-(len(trX) % self.BATCH_SIZE),...]
			self.trY 	= trY[:-(len(trY) % self.BATCH_SIZE),...]
			self.teX 	= teX[:-(len(teX) % self.BATCH_SIZE),...]
			self.teY 	= teY[:-(len(teY) % self.BATCH_SIZE),...]
			self.Ntr 	= len(self.trX)
			self.Nte 	= len(self.trX)
		else:
			self.trX 	= X
			self.trY 	= Y
			self.teX 	= None
			self.teY 	= None
			self.Ntr 	= len(self.trX)
			self.Nte  	= 0
		return self.trX , self.trY, self.teX, self.teY

	def BatchGenerator(self):
		X = np.zeros((self.BATCH_SIZE, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
		Y = np.zeros((self.BATCH_SIZE, self.input_shape[0], self.input_shape[1], self.nlabels), dtype = 'int')
		while True:
			for j in range(self.BATCH_SIZE):
				i = int(np.floor(np.random.uniform() * self.Ntr))
				X[j] = self.trX[i]
				Y[j] = self.trY[i]
			yield X, Y


	def train(self, STEP_SIZE, epochs = 5, saveto = None):
		if (self.teX is not None) and (self.teY is not None):
			#datagen 	= ImageDataGenerator(rotation_range = 10, width_shift_range = 0.1, height_shift_range = 0.1)
			datagen 	= ImageDataGenerator()
			datagen.fit(self.trX)
			self.model.fit_generator(	datagen.flow(self.trX, self.trY, batch_size = self.BATCH_SIZE), 
									 	steps_per_epoch = int(self.Ntr / self.BATCH_SIZE), 
									 	validation_data = (self.teX, self.teY), epochs = epochs )
		else:
			self.model.fit_generator(self.BatchGenerator(), steps_per_epoch = STEP_SIZE, epochs = epochs)

		if saveto is not None:
			print('saving model to ', saveto)
			self.model.save_weights(saveto)



