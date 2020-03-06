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

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def UNPOOL(pool, ind, ksize=[1, 2, 2, 1], name=None): # THIS FUNTION WORKS!!!!
    with tf.variable_scope('name') as scope:
        input_shape         = tf.shape(pool)
        output_shape        = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

        flat_input_size     = tf.cumprod(input_shape)[-1]
        flat_output_shape   = tf.stack([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])

        pool_               = tf.reshape(pool, tf.stack([flat_input_size]))
        batch_range         = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype = ind.dtype),
                                 		 shape = tf.stack([input_shape[0], 1, 1, 1]))

        b                   = tf.ones_like(ind) * batch_range
        b                   = tf.reshape(b, tf.stack([flat_input_size, 1]))

        ind_                = tf.reshape(ind, tf.stack([flat_input_size, 1]))
        ind_                = ind_ - b * tf.cast(flat_output_shape[1], tf.int64) #<-----
        ind_                = tf.concat([b, ind_], 1)

        ret                 = tf.scatter_nd(ind_, pool_, shape = tf.cast(flat_output_shape, tf.int64))
        ret                 = tf.reshape(ret, tf.stack(output_shape))

        set_input_shape     = pool.get_shape()
        set_output_shape    = [set_input_shape[0], set_input_shape[1] * ksize[1], set_input_shape[2] * ksize[2], set_input_shape[3]]
        ret.set_shape(set_output_shape)

    return ret



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



	'''
	def call(self, inputs):
		input_shape 	= K.cast(K.shape(inputs), dtype='int64')

		output_shape 	= (	input_shape[0],
							input_shape[1] * self.stride[1],
							input_shape[2] * self.stride[2],
							input_shape[3])

		#output_list = []
		#output_list.append(self.pooling_argmax // (output_shape[2] * output_shape[3]))
		#output_list.append(self.pooling_argmax % (output_shape[2] * output_shape[3]) // output_shape[3])
		argmax 			= self.pooling_argmax #K.stack(output_list)

		one_like_mask 	= K.ones_like(argmax)
		batch_range 	= K.reshape(K.arange(start=0, stop=input_shape[0], dtype='int64'), 
								shape=[input_shape[0], 1, 1, 1])

		b 				= one_like_mask * batch_range
		y 				= argmax // (output_shape[2] * output_shape[3])
		x 				= argmax % (output_shape[2] * output_shape[3]) // output_shape[3]
		feature_range 	= K.arange(start=0, stop=output_shape[3], dtype='int64')
		f 				= one_like_mask * feature_range

		# transpose indices & reshape update values to one dimension
		updates_size 	= tf.size(inputs)
		indices 		= K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
		values 			= K.reshape(inputs, [updates_size])
		return tf.scatter_nd(indices, values, output_shape)
	'''


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
		self.input_shape 	= (28, 28, 1)
		self.nlabels 		= 10 + 1 # 0 for background

	def predict(self, image):
		print(image.shape)
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
		if use_cpu:
			device = '/cpu:0'
		else:
			device = '/gpu:0'

		with tf.device(device):			
			if batch_size is not None:
				self.BATCH_SIZE = batch_size
				inputs = Input(shape = self.input_shape, batch_size = self.BATCH_SIZE)
			else:
				self.BATCH_SIZE = None
				inputs = Input(shape = self.input_shape)


			conv_block_1 = self.buildConv2DBlock(inputs, 32, 1, 2) # [?, 28, 28, 16]
			pool1, pool1_argmax = Lambda(self.max_pool_with_argmax, name='pool1')(conv_block_1) # [?, 14, 14, 16]

			conv_block_2 = self.buildConv2DBlock(pool1, 64, 2, 2)  # [?, 14, 14, 32]
			pool2, pool2_argmax = Lambda(self.max_pool_with_argmax, name='pool2')(conv_block_2) #[?, 7, 7, 32]

			fc3 = Conv2D(512, 7, use_bias=False, padding='valid', name='fc3')(pool2)    #[?, 1, 1, 512]
			fc3 = BatchNormalization(name='batchnorm_fc3')(fc3) 						#[?, 1, 1, 512]
			fc3 = Activation('relu', name='relu_fc3')(fc3)								#[?, 1, 1, 512]

			
			fc4 = Conv2D(512, 1, use_bias=False, padding='valid', name='fc4')(fc3)   	#[?, 1, 1, 512]
			fc4 = BatchNormalization(name='batchnorm_fc4')(fc4) 						#[?, 1, 1, 512]
			fc4 = Activation('relu', name='relu_fc4')(fc4)								#[?, 1, 1, 512]

			x 	= Conv2DTranspose(64, 7, use_bias=False, padding='valid', name='deconv-fc3')(fc4) 	#[?, 7, 7, 32]
			x 	= BatchNormalization(name='batchnorm_deconv-fc3')(x) 								#[?, 7, 7, 32]
			x 	= Activation('relu', name='relu_deconv-fc3')(x)            							#[?, 7, 7, 32]

			x 	= MaxUnpoolWithArgmax(pool2_argmax, name='unpool2')(x)  							#[?, 14, 14, 32]
			#x.set_shape(conv_block_2.get_shape())


			x = Conv2DTranspose(64, 3, use_bias=False, padding='same', name='deconv2-1')(x) 		#[?, 14, 14, 32]
			x = BatchNormalization(name='batchnorm_deconv2-1')(x) 									#[?, 14, 14, 32]
			x = Activation('relu', name='relu_deconv2-1')(x)  										#[?, 14, 14, 32]

			x = Conv2DTranspose(32, 3, use_bias=False, padding='same', name='deconv2-2')(x) 		#[?, 14, 14, 16]
			x = BatchNormalization(name='batchnorm_deconv2-2')(x) 									#[?, 14, 14, 16]
			x = Activation('relu', name='relu_deconv2-2')(x)   										#[?, 14, 14, 16]

			#x = Conv2DTranspose(16, 3, use_bias=False, padding='same', name='deconv2-3')(x) 		#[?, 14, 14, 16]
			#x = BatchNormalization(name='batchnorm_deconv2-3')(x) 									#[?, 14, 14, 16]
			#x = Activation('relu', name='relu_deconv2-3')(x)   									#[?, 14, 14, 16]

			x 	= MaxUnpoolWithArgmax(pool1_argmax, name='unpool1')(x) 								#[?, 28, 28, 16]
			#x.set_shape(conv_block_1.get_shape())

			x = Conv2DTranspose(32, 3, use_bias=False, padding='same', name='deconv1-1')(x) 		#[?, 28, 28, 16]
			x = BatchNormalization(name='batchnorm_deconv1-1')(x) 									#[?, 28, 28, 16]
			x = Activation('relu', name='relu_deconv1-1')(x)   										#[?, 28, 28, 16]

			x = Conv2DTranspose(32, 3, use_bias=False, padding='same', name='deconv1-2')(x) 		#[?, 28, 28, 16]
			x = BatchNormalization(name='batchnorm_deconv1-2')(x)	 								#[?, 28, 28, 16] 
			x = Activation('relu', name='relu_deconv1-2')(x)   										#[?, 28, 28, 16]

			#x = Conv2DTranspose(16, 3, use_bias=False, padding='same', name='deconv1-3')(x) 		#[?, 28, 28, 16]
			#x = BatchNormalization(name='batchnorm_deconv1-3')(x)	 								#[?, 28, 28, 16] 
			#x = Activation('relu', name='relu_deconv1-3')(x)   									#[?, 28, 28, 16]

			output = Conv2DTranspose(self.nlabels, 1, activation='softmax', padding='same', name='output')(x)

			self.model = Model(inputs=inputs, outputs=output)

			if print_summary:
				print(self.model.summary())

			self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse'])	

			return self.model

		
	'''
	def build_model_simple(self, use_cpu=True, print_summary=True, batch_size = None):
		print('building network')
		device = '/cpu:0'

		#with tf.device(device):			
		if batch_size is not None:
			self.BATCH_SIZE = batch_size
			inputs 			= Input(shape = self.input_shape, batch_size = self.BATCH_SIZE)
		else:
			self.BATCH_SIZE = None
			inputs = Input(shape = self.input_shape)



		conv_block_1 = self.buildConv2DBlock(inputs, 16, 1, 2)
		#pool1, pool1_argmax = Lambda(self.max_pool_with_argmax, name='pool1')(conv_block_1) 



		pool1 	= tf.nn.max_pool(	conv_block_1,
									ksize=[1,2,2,1],
									strides=[1,2,2,1],
									padding='SAME',
									name='pool1')

		input_shape 	= conv_block_1.get_shape().as_list()
		print(input_shape)
		mask_shape 		= (input_shape[0], int(input_shape [1]/2) ,int(input_shape[2]/2), input_shape[3])

		pool1_argmax 	= tf.zeros(mask_shape, dtype = tf.int64)

		for n in range(mask_shape[0]):
			for i in range(mask_shape[1]):
				for j in range(mask_shape[2]):
					in_indices = [ [n, w, h] for w in range(i*2, i*2+2) for h in range(j*2, j*2+2)]
					slices = tf.gather_nd(conv_block_1, in_indices)
					argmax = tf.argmax(slices, axis=0)
					indices_location 	= [[n, i, j, d] for d in range(input_shape[3])]
					sparse_indices 		= tf.SparseTensor(indices=indices_location, values=argmax, dense_shape=mask_shape)
					pool1_argmax 		= tf.sparse_add(pool1_argmax, sparse_indices)


		fc3 = Conv2D(512, 14, use_bias=False, padding='valid', name='fc3')(pool1) 
		fc3 = BatchNormalization(name='batchnorm_fc3')(fc3)
		fc3 = Activation('relu', name='relu_fc3')(fc3)			

		
		fc4 = Conv2D(512, 1, use_bias=False, padding='valid', name='fc4')(fc3)   
		fc4 = BatchNormalization(name='batchnorm_fc4')(fc4)
		fc4 = Activation('relu', name='relu_fc4')(fc4)

		x 	= Conv2DTranspose(16, 14, use_bias=False, padding='valid', name='deconv-fc3')(fc4)
		x 	= BatchNormalization(name='batchnorm_deconv-fc3')(x)
		x 	= Activation('relu', name='relu_deconv-fc3')(x)            

		x 	= MaxUnpoolWithArgmax(pool1_argmax, name='unpool2')(x)
		x.set_shape(conv_block_1.get_shape())


		x = Conv2DTranspose(32, 3, use_bias=False, padding='same', name='deconv2-1')(x)
		x = BatchNormalization(name='batchnorm_deconv2-1')(x)
		x = Activation('relu', name='relu_deconv2-1')(x)  

		x = Conv2DTranspose(32, 3, use_bias=False, padding='same', name='deconv2-2')(x)
		x = BatchNormalization(name='batchnorm_deconv2-2')(x)
		x = Activation('relu', name='relu_deconv2-2')(x)  

		output = Conv2DTranspose(self.nlabels, 1, activation='softmax', padding='same', name='output')(x)



		self.model = Model(inputs=inputs, outputs=output)

		if print_summary:
			print(self.model.summary())

		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse'])	

		return self.model
	'''

		
	def read_train_data(self, path_train, trainfolder = '', segfolder = '', train_sample_ratio = None, dtype = 'float', ):
		imagePaths 		= list(paths.list_images(path_train + trainfolder))

		X 			= np.empty((len(imagePaths), self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype = dtype)
		Y 			= np.empty((len(imagePaths), self.input_shape[0], self.input_shape[1], self.nlabels), dtype = 'int')

		for (i, path) in enumerate(imagePaths):
			img = cv2.imread(path)
			
			if self.input_shape[2] == 1:
				im 		= np.array(img[:,:,0], dtype = dtype) 	# shape =  (28, 28)
				im 		= np.expand_dims(im, axis = 2) 			# shape =  (28, 28, 1)
			else:
				im 		= np.array(image, dtype = dtype) 	# shape =  (28, 28)
			X[i] = im / 255.

			path_split 		= path.split(trainfolder)[0] + segfolder + path.split(trainfolder)[1]
			pathseg 		= path_split.split('.jpg')[0]  + '.png'

			print(path)
			print(pathseg)
			
			# cv2.imwrite(pathseg, img)
			img = cv2.imread(pathseg)

			im 		= np.array(img[:,:,0], dtype = 'int') 	# shape =  (28, 28)

			Y[i] 	= (np.arange(self.nlabels) == im[..., None]).astype(int)

		if train_sample_ratio is not None:
			Ntot 	= len(imagePaths)
			trX 	= X[:int(Ntot * train_sample_ratio),...]
			trY 	= Y[:int(Ntot * train_sample_ratio),...]
			teX 	= X[ int(Ntot * train_sample_ratio):,...]
			teY 	= Y[ int(Ntot * train_sample_ratio):,...]

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
			datagen 	= ImageDataGenerator(rotation_range = 10, width_shift_range = 0.2, height_shift_range = 0.2)
			datagen.fit(self.trX)



			# self.model.fit_generator(	datagen.flow(self.trX, self.trY, batch_size = self.BATCH_SIZE), 
									 	# steps_per_epoch = int(self.Ntr / self.BATCH_SIZE), 
									 	# validation_data = (self.teX, self.teY), epochs = epochs )
		else:
			self.model.fit_generator(self.BatchGenerator(), steps_per_epoch = STEP_SIZE, epochs = epochs)

		if saveto is not None:
			print('saving model to ', saveto)
			self.model.save_weights(saveto)



