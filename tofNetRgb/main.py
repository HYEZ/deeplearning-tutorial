from __future__ import absolute_import 	# absolute import 
from __future__ import division 		# make 1/2 = 0.5 happen. 1/2 = 0 for python2 without this declaration.
from __future__ import print_function	# allow only print function of python ver. 3/


import numpy as np 
import os 
#from imutils import paths
import cv2


from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras import utils
from tensorflow.keras import optimizers

from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model.facedeconv import DeconvNet



HEIGHT	 	= 224
WIDTH  		= 224
CHANNEL 	= 1

BATCH_SIZE  = 32


path_train	= '/Users/inchanji/Research/faceData/faceclip/train/'
path_model  = '/Users/inchanji/GoogleDrive/cnn/mymodel/tofFaceNet/model/facetNet_deconv.hdf5'


def train():
	STEPS_PER_EPOCH = 100

	deconv = DeconvNet()
	model  = deconv.build_model(print_summary = True, batch_size = BATCH_SIZE)

	trX, trY, teX, teY  = deconv.read_train_data(	path_train,  
													train_sample_ratio = 0.90, 
													trainfolder = 'image', 
													segfolder 	= 'seg')
	print('TRAIN DATA X:', np.shape(trX))
	print('TRAIN DATA Y:', np.shape(trY))
	print('TEST DATA X:', np.shape(teX))
	print('TEST DATA Y:', np.shape(teY))

	deconv.train(STEPS_PER_EPOCH, epochs = 50, saveto = path_model)


if __name__ == '__main__':

	train()


