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

def read_mnist_data(paths, label, one_hot = True, dtype = 'float32'):
	imgs 	= np.empty((len(paths), HEIGHT,WIDTH, CHANNEL), dtype=dtype)
	if one_hot:
		tgs 	= np.empty((len(paths), len(label)))	
	else:
		tgs 	= np.empty((len(paths), 1))

	for (i, path) in enumerate(paths):
		print("[INFO] processing image {}/{}".format(i + 1, len(paths)))
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



def read_mnist_data_onehot(path_train, path_truth):
	imagePaths 		= list(paths.list_images(path_train))
	truethPaths 	= list(paths.list_images(path_truth))

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

		truths[i] 	= (np.arange(self.nlabels) == im[..., None]).astype(int)

	return images, truths	



def train():
	imagePaths 		= list(paths.list_images(path))
	#for name in imagePaths: print(name)
	X, Y = read_mnist_data(imagePaths, LABEL)

	Ntot 	= len(X); 
	RATIO 	= 0.66
	trX 	= X[:int(Ntot*RATIO),...] # train
	teX 	= X[int(Ntot*RATIO):,...] # test
	trY 	= Y[:int(Ntot*RATIO),...]
	teY 	= Y[int(Ntot*RATIO):,...]
	
	# this is for fixing batch size!!! 
	trX = trX[:-(len(trX) % BATCH_SIZE),...]
	trY = trY[:-(len(trY) % BATCH_SIZE),...]
	teX = teX[:-(len(teX) % BATCH_SIZE),...]
	teY = teY[:-(len(teY) % BATCH_SIZE),...]


	datagen 	= ImageDataGenerator()
	datagen.fit(trX)

	if FIXED_BATCH_SIZE:
		mymodel = resNet(len(LABEL), BATCH_SIZE)
	else:
		mymodel = resNet(len(LABEL), None)

	mymodel.compile(optimizer=optimizers.Adam(lr=0.001), loss='categorical_crossentropy',metrics=['accuracy'])
	mymodel.summary()


	mymodel.fit_generator(	datagen.flow(trX, trY, batch_size = BATCH_SIZE),
							validation_data = (teX, teY), 
						 	steps_per_epoch = int(len(trX) / BATCH_SIZE), epochs=10)

	mymodel.save(path_model)



def inference():
	mymodel	 	= models.load_model(path_model)

	imgInfer 	= np.zeros((BATCH_SIZE, HEIGHT, WIDTH, CHANNEL))

	SingleTest = False

	size = 29
	if SingleTest:
		path_test  = '/mnt/c/Users/Z440_user1/Desktop/dataset/mnistNet/data/testSample/img_15.jpg'
		
		img 		= cv2.imread(path_test)
		img 		= np.array(img[:,:,0], dtype = 'float32')
		img 		= cv2.resize(img, (HEIGHT,WIDTH), interpolation = cv2.INTER_AREA)
		img 		/= 255
		img 		= np.expand_dims(img, axis=2)
		img 		= np.expand_dims(img, axis=0)	

		imgInfer[0,...] = img 

	else:
		for i in range(size):

			path_test 	= '/mnt/c/Users/Z440_user1/Desktop/dataset/mnistNet/data/testSample/img_'+str(i+100)+'.jpg'
			
			img 		= cv2.imread(path_test)
			img 		= np.array(img[:,:,0], dtype = 'float32')
			img 		= cv2.resize(img, (HEIGHT,WIDTH), interpolation = cv2.INTER_AREA)
			img 		/= 255
			img 		= np.expand_dims(img, axis=2)
			img 		= np.expand_dims(img, axis=0)	


			imgInfer[i,...] = img 

	print(imgInfer.shape)
	
	pred 	= mymodel.predict(imgInfer)

	for i in range(size):
		print(i+1, LABEL[np.argmax(pred[i])])



def conv(N_LABELS,  batch_size = None, dtype='float32'):
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
    x = _x

    _x = layers.Conv2D(	128, 3,
                        strides 	= (1, 1),
                        padding 	= 'same',
                        activation 	= 'relu',
                        use_bias 	= True,
                        bias_initializer 	= 'zeros',
                        kernel_initializer 	= 'he_normal')(_x) 

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