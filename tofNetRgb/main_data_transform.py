from __future__ import absolute_import 	# absolute import 
from __future__ import division 		# make 1/2 = 0.5 happen. 1/2 = 0 for python2 without this declaration.
from __future__ import print_function	# allow only print function of python ver. 3/
import copy
import numpy as np 
import os 
from imutils import paths
import cv2
import pprint


tgtname  = '132824'

nrow = 480
ncol = 640

maxval 	= 4095
fx 		= lambda x: maxval if x < 0 else x


path 	= '/mnt/c/Users/Z440_user1/Desktop/dataset/tofNetRgb_faceData/'
pathout 	= '/mnt/c/Users/Z440_user1/Desktop/dataset/tofNetRgb_faceData/transform'
# path   	 = '/Users/inchanji/Research/faceData/VAP_RGBD_FaceData/' + tgtname + '/'
# pathout  = '/Users/inchanji/Research/faceData/VAP_RGBD_FaceDataReduced/' + tgtname + '/'

if not os.path.isdir(pathout):
    os.mkdir(pathout)
    

imglist  = list(paths.list_images(path))

for filename0 in imglist:
	try:
		imgfilename = filename0.split(os.sep)[-1].split('_c')[0]
		

		imgname  	= imgfilename + '_c.bmp'
		distname 	= imgfilename + '_d.dat'
		distnameBMP = imgfilename + '_d.bmp'

		print(imgname, distname, distnameBMP)

		img 	= cv2.imread(path + imgname)[...,::-1]
		depth 	= np.zeros((nrow, ncol))

		with open(path + distname, 'r') as f:
			for i, line in enumerate(f):
				for j, d in enumerate(line.split()):
					depth[i,j]  = fx(int(d))	

		
		imgresize = cv2.resize(img, (ncol, nrow), interpolation=cv2.INTER_CUBIC)
		depthnew  = (np.roll(depth, 15, axis = 1 ) / 4096 * 255).astype('int')
		depthnew  = np.roll(depthnew, -5, axis = 0)

		cv2.imwrite(pathout + imgname, imgresize[:,:,::-1])
		cv2.imwrite(pathout + distnameBMP, depthnew)
	except:
		print("no")
		pass