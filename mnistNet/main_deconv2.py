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
from model.deconv import MaxPoolingWithArgmax2D,MaxUnpooling2D

from model.newdeconv import DeconvNet

L2_WEIGHT_DECAY 	= 1e-4
BATCH_NORM_DECAY 	= 0.9
BATCH_NORM_EPSILON = 1e-5

HEIGHT	 	= 28
WIDTH  		= 28
CHANNEL 	= 1

OFFSET 		= 1

BATCH_SIZE 	= 32

root = '/mnt/c/Users/Z440_user1/Desktop/dataset/mnistNet'

path_train	= root + '/data/'
path_model  = root + '/mnistNet_deconv.hdf5'
LABEL 		= np.array(['0','1','2','3','4','5','6','7','8','9'])


FIXED_BATCH_SIZE = True

def read_mnist_data(paths, label, one_hot = True, dtype = 'float32'):
	imgs 	= np.empty((len(paths), HEIGHT,WIDTH, CHANNEL), dtype=dtype)
	if one_hot:
		tgs 	= np.empty((len(paths), len(label)))	
	else:
		tgs 	= np.empty((len(paths), 1))

	for (i, path) in enumerate(paths):
		#print("[INFO] processing image {}/{}".format(i + 1, len(paths)))
		name  	= path.split(os.path.sep)#[-2]
		image 	= cv2.imread(path)


		if CHANNEL == 1:
			im 		= np.array(image[:,:,0], dtype = dtype) 	# shape =  (28, 28)
			im 		= np.expand_dims(im, axis = 2) 				# shape =  (28, 28, 1)
		else:
			im 		= np.array(image, dtype = dtype) 	# shape =  (28, 28)

		
		idx = np.where(LABEL == name[-2])[0][0]
		if one_hot:
			tgs[i,:] 	= 0.
			tgs[i,idx] 	= 1.
		else:
			
			tgs[i] = idx


		imgs[i] = im
		#print(path)
		#print(name[-2], np.shape(im), np.shape(image))
		#print(tgs[i])

	index 	= np.random.permutation(imgs.shape[0])
	imgs 	= imgs[index]
	tgs 	= tgs[index]

	if dtype == 'float32': imgs /= 255.
	return imgs, tgs



def read_mnist_data_onehot(path_train):
	imagePaths 		= list(paths.list_images(path_train))

	images 			= np.empty((len(imagePaths), HEIGHT, WIDTH, CHANNEL), dtype = 'float')
	truths 			= np.empty((len(imagePaths), HEIGHT, WIDTH, 11), dtype = 'int')

	for (i, path) in enumerate(imagePaths):
		img = cv2.imread(path)
		
		im 		= np.array(img[:,:,0], dtype =  'float') 	# shape =  (28, 28)
		im 		= np.expand_dims(im, axis = 2) 			# shape =  (28, 28, 1)

		images[i] 	= im / 255.

		path_split 	= path.split('trainingSample')
		path 		= path_split[0] + 'trainingSampleSeg' + path_split[1].split('.jpg')[0] + '.png'
		
		img 		= cv2.imread(path)

		im 			= np.array(img[:,:,0], dtype = 'int') 	# shape =  (28, 28)

		truths[i] 	= (np.arange(11) == im[..., None]).astype(int)

	return images, truths	




def vggdeconv(N_LABELS,  batch_size = None, dtype='float32'):
	input_shape = (HEIGHT, WIDTH, CHANNEL)
	convdepth 	= [16, 32, 64, 128]
	deconvdepth = convdepth[::-1].copy()
	fcdepth 	= [256]
	BNaxis 		= 3	


	img_input = layers.Input( 	shape = input_shape, 
								dtype = dtype, 
								batch_size = batch_size)	

	# Conv Layer 1, 28 x 28
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

	x = layers.Conv2D(	convdepth[0], (3, 3),
						strides 	= (1, 1),
						padding 	= 'same', 
						use_bias 	= True,
						bias_initializer 	= 'zeros',
						kernel_initializer 	= 'he_normal',
						#kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
						name='conv1_2')(img_input) 


	x = layers.BatchNormalization(	axis 	 = BNaxis,
									momentum = BATCH_NORM_DECAY,
									epsilon  = BATCH_NORM_EPSILON,
									name 	 = 'bn1_2')(x)		

	x = layers.Activation('relu')(x)				

	x, mask1 = MaxPoolingWithArgmax2D()(x) # [28, 28, 8] -> [14, 14, 8]

	shape2 = tuple(x.get_shape().as_list()[1:3])

	# Conv Layer 2, 14 x 14
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

	x = layers.Conv2D(	convdepth[1], (3, 3),
						strides 	= (1, 1),
						padding 	= 'same', 
						use_bias 	= True,
						bias_initializer 	= 'zeros',
						kernel_initializer 	= 'he_normal',
						#kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
						name='conv2_2')(x) 


	x = layers.BatchNormalization(	axis 	 = BNaxis,
									momentum = BATCH_NORM_DECAY,
									epsilon  = BATCH_NORM_EPSILON,
									name 	 = 'bn2_2')(x)		

	x = layers.Activation('relu')(x)		

	#x = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
	x, mask2 = MaxPoolingWithArgmax2D()(x)

	# Conv Layer 3, 7 x 7

	shape3 = tuple(x.get_shape().as_list()[1:3])
	x = layers.Conv2D(	fcdepth[0], shape3,
						strides 	= (1, 1),
						padding 	= 'valid', 
						use_bias 	= True,
						bias_initializer 	= 'zeros',
						kernel_initializer 	= 'he_normal',
						#kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
						name='conv3')(x) 


	x = layers.BatchNormalization(	axis 	 = BNaxis,
									momentum = BATCH_NORM_DECAY,
									epsilon  = BATCH_NORM_EPSILON,
									name 	 = 'bn3')(x)		

	x = layers.Activation('relu')(x)				
	
	shape4 = tuple(x.get_shape().as_list()[1:3])

	# FC Layer 5, 1 x 1
	x = layers.Conv2D(	fcdepth[0], shape4,
						strides 	= (1, 1),
						padding 	= 'valid', 
						use_bias 	= True,
						bias_initializer 	= 'zeros',
						kernel_initializer 	= 'he_normal',
						name='fc1')(x) 

	x = layers.BatchNormalization(	axis 	 = BNaxis,
									momentum = BATCH_NORM_DECAY,
									epsilon  = BATCH_NORM_EPSILON,
									name 	 = 'bn4')(x)		

	x = layers.Activation('relu')(x)

	# Deconv Layer 1, 1 x 1
	x = layers.Conv2DTranspose(	
					deconvdepth[2], shape3,
					strides 	= (1, 1),
					padding 	= 'valid', 
					bias_initializer 	= 'zeros',
					kernel_initializer 	= 'he_normal',
					kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
					name='deconv1')(x)

	x = layers.BatchNormalization(	axis 	 = BNaxis,
									momentum = BATCH_NORM_DECAY,
									epsilon  = BATCH_NORM_EPSILON,
									name 	 = 'bn5')(x)		
	x = layers.Activation('relu')(x)


	# Unpooling 1
	x =	MaxUnpooling2D()([x, mask2])

	# Deconv Layer 2, 14 x 14
	x = layers.Conv2DTranspose(	
					deconvdepth[2], (3,3),
					strides 	= (1, 1),
					padding 	= 'same', 
					bias_initializer 	= 'zeros',
					kernel_initializer 	= 'he_normal',
					kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
					name='deconv2_1')(x)

	x = layers.BatchNormalization(	axis 	 = BNaxis,
									momentum = BATCH_NORM_DECAY,
									epsilon  = BATCH_NORM_EPSILON,
									name 	 = 'bn6')(x)		

	x = layers.Activation('relu')(x)


	x = layers.Conv2DTranspose(	
					deconvdepth[3], (3,3),
					strides 	= (1, 1),
					padding 	= 'same', 
					bias_initializer 	= 'zeros',
					kernel_initializer 	= 'he_normal',
					kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
					name='deconv2_2')(x)

	x = layers.BatchNormalization(	axis 	 = BNaxis,
									momentum = BATCH_NORM_DECAY,
									epsilon  = BATCH_NORM_EPSILON,
									name 	 = 'bn7')(x)		

	x = layers.Activation('relu')(x)		

	# Unpooling 2
	x =	MaxUnpooling2D()([x, mask1]) # []

	x = layers.Conv2DTranspose(	
					deconvdepth[3], (3,3),
					strides 	= (1, 1),
					padding 	= 'same', 
					bias_initializer 	= 'zeros',
					kernel_initializer 	= 'he_normal',
					kernel_regularizer 	= regularizers.l2(L2_WEIGHT_DECAY),
					name='deconv3')(x)

	x = layers.BatchNormalization(	axis 	 = BNaxis,
									momentum = BATCH_NORM_DECAY,
									epsilon  = BATCH_NORM_EPSILON,
									name 	 = 'bn8')(x)		

	x = layers.Activation('relu')(x)	

	x = backend.cast(x, 'float32')
	x = layers.Conv2DTranspose(11, 1, activation='softmax', padding='same', name='output')(x)


	return models.Model(img_input, x, name = 'vggdeconv')





def train():
	#for name in imagePaths: print(name)
	X, Y 		= read_mnist_data_onehot(pathtrain)

	#print(np.shape(X[0]))
	#print(np.shape(Y[0]))

	Ntot 	= len(X); 
	RATIO 	= 0.66
	trX 	= X[:int(Ntot*RATIO),...]
	teX 	= X[int(Ntot*RATIO):,...]
	trY 	= Y[:int(Ntot*RATIO),...]
	teY 	= Y[int(Ntot*RATIO):,...]
	
	# this is for fixing batch size!!!
	trX 	= trX[:-(len(trX) % BATCH_SIZE),...]
	trY 	= trY[:-(len(trY) % BATCH_SIZE),...]
	teX 	= teX[:-(len(teX) % BATCH_SIZE),...]
	teY 	= teY[:-(len(teY) % BATCH_SIZE),...]

	
	datagen 	= ImageDataGenerator()
	datagen.fit(trX)

	if FIXED_BATCH_SIZE:
		mymodel = vggdeconv(len(LABEL), BATCH_SIZE)
	else:
		mymodel = vggdeconv(len(LABEL), None)

	mymodel.compile(optimizer=optimizers.Adam(lr=0.001), loss='categorical_crossentropy',metrics=['accuracy','mse'])
	mymodel.summary()
	

	mymodel.fit_generator(	datagen.flow(trX, trY, batch_size = BATCH_SIZE),
							validation_data = (teX, teY), 
						 	steps_per_epoch = int(len(trX) / BATCH_SIZE), epochs=10)

	mymodel.save(path_model)
	


def inference():
	mymodel	 	= models.load_model(path_model)

	imgInfer 	= np.zeros((BATCH_SIZE, HEIGHT, WIDTH, CHANNEL))

	SingleTest = False

	if SingleTest:
		path_test 	= root + '/data/testSample/img_106.jpg'
		
		img 		= cv2.imread(path_test)
		img 		= np.array(img[:,:,0], dtype = 'float32')
		img 		= cv2.resize(img, (HEIGHT,WIDTH), interpolation = cv2.INTER_AREA)
		img 		/= 255
		img 		= np.expand_dims(img, axis=2)
		img 		= np.expand_dims(img, axis=0)	

		imgInfer[0,...] = img 

	else:
		for i in range(BATCH_SIZE):

			path_test 	= root + '/data/testSample/img_'+str(i+100)+'.jpg'
			
			img 		= cv2.imread(path_test)
			img 		= np.array(img[:,:,0], dtype = 'float32')
			img 		= cv2.resize(img, (HEIGHT,WIDTH), interpolation = cv2.INTER_AREA)
			img 		/= 255
			img 		= np.expand_dims(img, axis=2)
			img 		= np.expand_dims(img, axis=0)	

			imgInfer[i,...] = img 

	
	pred 	= mymodel.predict(imgInfer)

	for i in range(BATCH_SIZE):
		#print(pred[i])
		print(i+1, LABEL[np.argmax(pred[i])])



def inference_deconv2(modelnet):
	BATCH_SIZE  	= 32
	imgInfer 	= np.zeros((BATCH_SIZE, HEIGHT, WIDTH, CHANNEL))

	path_test 	= root + '/data/testSample/img_106.jpg'
	
	img 		= cv2.imread(path_test)
	img 		= np.array(img[:,:,0], dtype = 'float32')
	img 		= cv2.resize(img, (HEIGHT,WIDTH), interpolation = cv2.INTER_AREA)
	img 		/= 255
	img 		= np.expand_dims(img, axis=2)
	img 		= np.expand_dims(img, axis=0)	

	imgInfer[0,...] = img 

	print(np.shape(imgInfer))
	pred 	= modelnet.predict(imgInfer)
	print(pred)


	pred 	= pred[0,...]
	print(np.shape(pred))
	out 	= np.zeros((HEIGHT, WIDTH), dtype=np.int)
	for i in range(HEIGHT):
		for j in range(WIDTH):

			idx 	= int(np.argmax(pred[i,j,:]))
			out[i,j] = idx

	print(out.ravel())



def train_deconv():
	STEPS_PER_EPOCH = 100

	deconv = DeconvNet()
	model  = deconv.build_model(print_summary = True , batch_size = BATCH_SIZE)

	trX, trY, teX, teY  = deconv.read_train_data(	path_train,  
													train_sample_ratio = 0.66, 
													trainfolder = 'trainingSet', 
													segfolder 	= 'trainingSetSeg')
	print('TRAIN DATA X:', np.shape(trX))
	print('TRAIN DATA Y:', np.shape(trY))
	print('TEST DATA X:', np.shape(teX))
	print('TEST DATA Y:', np.shape(teY))


	#datagen 	= ImageDataGenerator()
	#datagen.fit(trX)

	#model.fit_generator(	datagen.flow(trX, trY, batch_size = BATCH_SIZE),
	#					 	steps_per_epoch = int(len(trX) / BATCH_SIZE), epochs=10)

	deconv.train(STEPS_PER_EPOCH, epochs = 20, saveto = path_model)

	
	#deconv.save(path_model)
	#return deconv




def inference_deconv():
	deconv 		= DeconvNet()
	deconv.build_model(print_summary = True, batch_size = BATCH_SIZE)
	deconv.load(path_model)


	imgInfer 	= np.zeros((BATCH_SIZE, HEIGHT, WIDTH, CHANNEL))

	path_test 	= root + '/data/trainingSample/6/img_74.jpg'
	
	img 		= cv2.imread(path_test)
	img 		= np.array(img[:,:,0], dtype = 'float32')
	img 		= cv2.resize(img, (HEIGHT,WIDTH), interpolation = cv2.INTER_AREA)
	img 		/= 255
	img 		= np.expand_dims(img, axis=2)
	img 		= np.expand_dims(img, axis=0)	

	for i in range(BATCH_SIZE):
		imgInfer[i] = img 

	print(np.shape(imgInfer))
	pred 	= deconv.predict(imgInfer)
	

	pred 	= pred[0]

	out 	= np.zeros((HEIGHT, WIDTH), dtype = np.int)
	for i in range(HEIGHT):
		for j in range(WIDTH):
			idx 	= int(np.argmax(pred[i,j,:]))
			out[i,j] = idx

	print(out)




if __name__ == '__main__':

	#train()
	#inference()


	# train_deconv()
	inference_deconv()
