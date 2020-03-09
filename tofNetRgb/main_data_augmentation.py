from __future__ import absolute_import 	# absolute import 
from __future__ import division 		# make 1/2 = 0.5 happen. 1/2 = 0 for python2 without this declaration.
from __future__ import print_function	# allow only print function of python ver. 3/
import copy
import numpy as np 
import os 
from imutils import paths
import cv2
import ntpath
import face_recognition
import face_segmentator
import time
import pprint


def filterImgList(imagePaths):
	imgListOut = []
	for imgfilename in imagePaths:
		filetype = imgfilename.split(os.sep)[-1].split('.')[0][-1]
		if filetype == 'c':
			imgListOut.append(imgfilename)
	return imgListOut

pathdata 	= '/mnt/c/Users/Z440_user1/Desktop/dataset/tofNetRgb_faceData/'
pathsaveto 	= '/mnt/c/Users/Z440_user1/Desktop/dataset/tofNetRgb_faceData/augmented'
Naugment 	=  10


if not os.path.isdir(pathsaveto):
	os.mkdir(pathsaveto)

imagePaths 	= list(paths.list_images(pathdata))
imagePaths 	= filterImgList(imagePaths)




Ntot 			= len(imagePaths)
Nerr 			= 0
#filename =  ntpath.basename(imagePaths[-1])
time0 = time.time()

for n, path in enumerate(imagePaths):
	#if n > 0: break
	print('{}/{} ({})'.format(n+1,Ntot, Nerr))
	time1 = time.time()

	elapsed = (time1 - time0)/ 60.
	est 	= elapsed / (n+1) * Ntot  - elapsed
	print('time elapsed:{:d} min., est {:d} min.'.format(int(elapsed), int(est)))


	try:
		img0 				= cv2.imread(path)[...,::-1]
		dep0 				= cv2.imread(path.split('_c.bmp')[0]+'_d.bmp')[...,0]
		image 				= face_recognition.load_image_file(path)
		face_landmarks_list = face_recognition.face_landmarks(image)
		attributeList       = face_segmentator.getLandmakrPts(face_landmarks_list)

	except:
		
		Nerr += Naugment
		print('Error while converting image ({})'.format(Nerr))		
		continue

					
	for i in range(Naugment):
		#try:
		imgname 	= ('00000' + str(n+1))[-5:]
		imgOutPath 	= pathsaveto + 'img_' + imgname + '_' + str(i+1) + '.png'
		segOutPath  = pathsaveto + 'seg_' + imgname + '_' + str(i+1) + '.bmp'
		distOutPath = pathsaveto + 'dep_' + imgname + '_' + str(i+1) + '.bmp'

		imgout, pts, segmentation, maskcolor, depthout \
				= face_segmentator.getModifiedSegmentation(	img0.copy(), 
															[1,2,3,4,5], 
															copy.deepcopy(attributeList), 
															dep0,
															facrange = (3,5))

		cv2.imwrite(imgOutPath, imgout[:,:,::-1])
		cv2.imwrite(segOutPath, segmentation)
		cv2.imwrite(distOutPath, depthout)
		
		print(imgOutPath)

		#except:
		#	Nerr += 1
		#	print('Error while converting image ({})'.format(Nerr))



