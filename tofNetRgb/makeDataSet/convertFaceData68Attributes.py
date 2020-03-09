from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

from PIL import Image
import numpy as np 
import os 
from imutils import paths
import cv2
import matplotlib.path
from scipy import ndimage


pathdata 	= '/Users/inchanji/Downloads/FACE_ATTRIBUTES/FULL/train/'
pathsaveto 	= '/Users/inchanji/Research/faceData/faceclip/train/'

def extract_pts(path):
	arr = []
	for i in open(path):
		try: 
			i = i.split('\n')[0].split(' ')
			arr.append([float(i[0]), float(i[1])])
		except:
			pass
	return np.array(arr)


def clipfaceimg(img, pts0, resize = None, fac = None):
	pts  	= pts0.copy()
	xpos 	= pts[:,0]
	ypos 	= pts[:,1]

	left    = np.min(xpos)
	right   = np.max(xpos)

	top     = np.max(ypos)
	bottom  = np.min(ypos)

	xcen    = 0.5 * (right + left)
	ycen    = 0.5 * (top + bottom)    

	width  	= right - left
	height 	= top - bottom

	x0 		= int(np.max([0, xcen - width  * fac[0] / 2]))
	x1 		= int(np.min([np.shape(img)[1], xcen + width * fac[0] / 2 + 1]))

	y0 		= int(np.max([0, ycen - height * fac[1] / 2]))
	y1 		= int(np.min([np.shape(img)[0], ycen + height * fac[1] / 2 + 1]))

	pts 	-= [x0, y0]

	imgclip = img[y0:y1,x0:x1].copy()

	if resize is not None:
		normxfac = resize[0] / (x1 - x0)
		normyfac = resize[1] / (y1 - y0)
		imgclip  = cv2.resize(imgclip.astype('float'), (resize[0], resize[1]), cv2.INTER_AREA) 

		pts[:,0] *= normxfac 
		pts[:,1] *= normyfac

	return imgclip.astype('int'), pts




def index_face_attribute():
	IND 			= []
	ind_face 		= np.arange(27,dtype=int)
	ind_face[17:] 	= ind_face[17:][::-1]
	IND.append(ind_face)
	ind_reye 		= np.linspace(36,41, 41-36+1, dtype=int)
	IND.append(ind_reye)
	ind_leye 		= np.linspace(42,47, 47-42+1, dtype=int)
	IND.append(ind_leye)
	ind_nose 		= np.linspace(31,35, 35-31+1, dtype=int)
	ind_nose 		= np.concatenate(([27],ind_nose))
	IND.append(ind_nose)
	ind_mouse 		= np.linspace(48,59, 59-48+1, dtype=int)
	IND.append(ind_mouse)
	return IND


def maskmap(MASKCOLOR, IND, pts):
	MASKS = []
	BNDRY = []
	for n in range(len(MASKCOLOR)):
		pos_x 		= pts[IND[n],0].copy()
		pos_y 		= pts[IND[n],1].copy()
		temp_list 	= []
		for a, b in zip(pos_x, pos_y):
			temp_list.append([a, b])

		polygon  	= np.array(temp_list)
		left     	= np.min(polygon, axis=0)
		right    	= np.max(polygon, axis=0)

		x 			= np.arange(np.ceil(left[0]), np.floor(right[0])+1)
		y 			= np.arange(np.ceil(left[1]), np.floor(right[1])+1)

		xv, yv  	= np.meshgrid(x, y, indexing='xy')
		points  	= np.hstack((xv.reshape((-1,1)), yv.reshape((-1,1))))

		path     	= matplotlib.path.Path(polygon)
		mask     	= path.contains_points(points)
		mask.shape 	= xv.shape

		x0 = int(yv[0,0])
		x1 = int(yv[0,0] + np.shape(mask)[0])
		y0 = int(xv[0,0])
		y1 = int(xv[0,0] + np.shape(mask)[1])

		MASKS.append(mask)
		BNDRY.append([x0,x1,y0,y1])
	return MASKS, BNDRY

def segmentation_map(img, pts0):
	labels 			= [ 1, 2, 3, 4, 5 ]  # [face, right eye, left eye, nose, mouse]
	label_rgbcolors = [ [0, 0, 255], [255, 0, 0], [255, 255, 0], [0, 255, 0],  [0, 255, 255] ] #  [Blue Red Yellow Green Cyan]

	index 			= index_face_attribute()
	if len(label_rgbcolors) != len(index): 
		return None

	masks, bndry 	= maskmap(labels, index, pts0)
	seg 			= np.zeros(np.shape(img)[:2], dtype=int)
	maskcolor  		= np.zeros(np.shape(img), dtype=int)

	for n in range(len(masks)):
		x0, x1, y0, y1 = bndry[n]
		
		x0 = int(np.max([0, x0]))
		y0 = int(np.max([0, y0]))

		x1 = int(np.min([x1, np.shape(img)[0]]))
		y1 = int(np.min([y1, np.shape(img)[1]]))

		for i in range(x0,x1):
			for j in range(y0,y1):
				if masks[n][i-x0,j-y0]:
					if (n == 1)  or (n == 2):
						seg[i-1:i+2,j-1:j+2]  		 =  labels[n]
						maskcolor[i-1:i+2,j-1:j+2,:] =  label_rgbcolors[n]
					else:
						seg[i,j]  		 	=  labels[n]
						maskcolor[i,j,:] 	=  label_rgbcolors[n]

	return seg, maskcolor	


def rotate(np_img, deg = 10):
	img = ndimage.rotate(np_img, deg, mode='wrap', reshape=False)
	return img 

def rotate_pts(pts, pos0, deg = 10):
	rad 	= np.pi / 180 * deg
	pts_rot = pts.copy()

	for i, pos in enumerate(pts):
		pts_rot[i,0] =  pos0[0] + np.cos(rad) * (pos[0] - pos0[0]) + np.sin(rad) * (pos[1] - pos0[1])
		pts_rot[i,1] =  pos0[1] + np.cos(rad) * (pos[1] - pos0[1]) - np.sin(rad) * (pos[0] - pos0[0])
	return pts_rot

def move_img_to_center(img, pts):
	xpts, ypts   = pts[:,0], pts[:,1]
	xpmed, ypmed = np.median(xpts), np.median(ypts)

	xshift = int(np.shape(img)[1]/2 - xpmed)
	yshift = int(np.shape(img)[0]/2 - ypmed)

	imgroll = np.roll(img, (yshift, xshift), axis = (0,1))

	pts_out = pts.copy()
	pts_out += [xshift, yshift]

	return imgroll, pts_out


def augment_sample(img0, pts0, maxdeg = 50, resize = (96, 96), facrange = (2, 4), maxshift = 0.5):
	labels          = [ 1, 2, 3, 4, 5 ]  # [face, right eye, left eye, nose, mouse]
	#label_rgbcolors = [ [0, 0, 255], [255, 0, 0], [255, 255, 0], [0, 255, 0],  [0, 255, 255] ] #  [Blue Red Yellow Green Cyan]

	deg 		= int(maxdeg * (np.random.uniform()*2 - 1)*np.random.uniform()) // 10 * 10 		# in unit of 10 degrees
	index 		= index_face_attribute()
	#if len(label_rgbcolors) != len(index): 
	#	return None

	fac 		= (facrange[0] + (facrange[1]-facrange[0]) * np.random.uniform()) // 0.5 * 0.5
	fac 		= (fac, fac)
	img, pts    = move_img_to_center(img0, pts0)
	img 		= rotate(img, deg = deg)
	pts 		= rotate_pts(pts, np.array(np.shape(img)[:2][::-1])/2., deg = deg)

	img, pts   	= clipfaceimg(img, pts, resize = resize, fac = fac)

	masks, bndry 	= maskmap(labels, index, pts)
	seg 			= np.zeros(np.shape(img)[:2], dtype=int)
	#maskcolor  	= np.zeros(np.shape(img), dtype=int)

	for n in range(len(masks)):
		x0, x1, y0, y1 = bndry[n]

		x0 	= int(np.max([0, x0]))
		y0 	= int(np.max([0, y0]))

		x1 	= int(np.min([x1, np.shape(img)[0]]))
		y1 	= int(np.min([y1, np.shape(img)[1]]))

		for i in range(x0,x1):
			for j in range(y0,y1):
				if masks[n][i-x0,j-y0]:
					if (n == 1)  or (n == 2):
						seg[i-1:i+2,j-1:j+2] = labels[n]
						#maskcolor[i-1:i+2,j-1:j+2,:] = label_rgbcolors[n]
					else:
						seg[i,j]         	 = labels[n]
						#maskcolor[i,j,:] = label_rgbcolors[n]

	xmin = np.min(pts[:,0]); xmax = np.max(pts[:,0]);
	ymin = np.min(pts[:,1]); ymax = np.max(pts[:,1]);

	shift_right_max = np.shape(img)[1] - int(np.ceil(xmax) + 1)
	shift_left_max  = - int(np.floor(xmin) - 1)

	shift_up_max    = np.shape(img)[0] - int(np.ceil(ymax) + 1)
	shift_down_max  = - int(np.floor(ymin) - 1) 

	xshift 			= int([shift_left_max, shift_right_max][int(np.random.uniform()*2)]  * maxshift) // 10 * 10
	yshift 			= int([shift_up_max, shift_down_max][int(np.random.uniform()*2)]  * maxshift) // 10 * 10   

	shift = [yshift, xshift]

	img = np.roll(img, shift, axis=(0,1))    
	seg = np.roll(seg, shift, axis=(0,1))    
	#maskcolor = np.roll(maskcolor, shift, axis=(0,1))        

	return img, pts + shift[::-1], seg#, maskcolor



def main():
	
	Naugment 		= 10
	imagePaths 		= list(paths.list_images(pathdata))
	Nerr = 0
	Ntot = len(imagePaths)
	for i, imgpath in enumerate(imagePaths):
		print('{}/{} ({})'.format(i+1,Ntot, Nerr))
		print(imgpath)
		try:
			ptspath = imgpath.split('.')[0] + '.pts'
			imgname = ('00000' + str(i))[-5:]

			img 	= cv2.imread(imgpath)
			pts 	= extract_pts(ptspath)

			#imgclip, ptsclip 			= clipfaceimg(img, pts, resize = (96, 96), fac = [2., 2.])
			#segclip, segclip_color 	= segmentation_map(imgclip, ptsclip)

			for j in range(Naugment):
				imgclip, ptsclip, segclip 	= augment_sample(img, pts, maxdeg = 40, resize = (96, 96), facrange = (2, 4), maxshift = 0.5)
				cv2.imwrite(pathsaveto+'image/' + imgname + '_' + str(j+1) + '.png', imgclip)
				cv2.imwrite(pathsaveto+'seg/'   + imgname + '_' + str(j+1) + '.png', segclip)

		except:
			Nerr += 1
			print('Error while converting image ({})'.format(Nerr))

		


if __name__ == '__main__':
	main()