import numpy as np 
import matplotlib.path
from scipy import ndimage
import cv2



def getModifiedSegmentation(imgPath, segPath, deg = None, shift = None, 
							maxdeg = None, resize = None, fac = None, maxshift = None):
	img0 = cv2.imread(imgPath)[...,::-1]
	seg0 = cv2.imread(segPath)[:,:,0]

	if maxdeg is None:  maxdeg = 50
	if maxshift is None: maxshift = 0.5
	if resize is None: resiae = (224, 224)
	facrange = (2,4)

	if deg is None:
		deg = int(maxdeg * (np.random.uniform()*2 - 1)*np.random.uniform()) // 10 * 10
	else:
		deg = int(deg)

	if fac is None:
		fac = (facrange[0] + (facrange[1]-facrange[0]) * np.random.uniform()) // 0.5 * 0.5
		fac = (fac, fac)
	else:
		fac = (fac, fac)

	pts, ptsRavel = getAttributePts(seg0)

	yc, xc = np.median(ptsRavel, axis = 0)

	img  = move_img_to_center(img0, xc, yc)
	seg  = move_img_to_center(seg0, xc, yc)

	img  = rotateImg(img, deg = deg)
	seg  = rotateSeg(seg, deg = deg)


	img = clipfaceimg(img, ptsRavel, resize = resize, fac = fac)
	seg = clipfaceimg(seg, ptsRavel, resize = resize, fac = fac)


	ymin, xmin = np.min(ptsRavel, axis = 0)
	ymax, xmax = np.max(ptsRavel, axis = 0)

	if shift is None:
		shift_right_max = np.shape(img)[1] - int(np.ceil(xmax) + 1)
		shift_left_max  = - int(np.floor(xmin) - 1)

		shift_up_max    = np.shape(img)[0] - int(np.ceil(ymax) + 1)
		shift_down_max  = - int(np.floor(ymin) - 1) 

		xshift	= int([shift_left_max, shift_right_max][int(np.random.uniform()*2)]  * maxshift) // 10 * 10
		yshift	= int([shift_up_max, shift_down_max][int(np.random.uniform()*2)]  * maxshift) // 10 * 10   
		shift 	= [yshift, xshift]

	img 	= np.roll(img, shift, axis=(0,1))    
	seg 	= np.roll(seg, shift, axis=(0,1))    	


	return img, seg.astype('int')
	 

def clipfaceimg(img, pts, resize = None, fac = None):
	
	bottom, left  = np.min(pts, axis = 0)
	top, right    = np.max(pts, axis = 0)

	xcen    = 0.5 * (right + left)
	ycen    = 0.5 * (top + bottom)    

	width  	= right - left
	height 	= top - bottom

	x0 		= int(np.max([0, xcen - width  * fac[0] / 2]))
	x1 		= int(np.min([np.shape(img)[1], xcen + width * fac[0] / 2 + 1]))

	y0 		= int(np.max([0, ycen - height * fac[1] / 2]))
	y1 		= int(np.min([np.shape(img)[0], ycen + height * fac[1] / 2 + 1]))

	imgclip = img[y0:y1,x0:x1].copy()

	if resize is not None:
		normxfac = resize[1] / (x1 - x0)
		normyfac = resize[0] / (y1 - y0)
		imgclip  = cv2.resize(imgclip.astype('float'), (resize[0], resize[1]), cv2.INTER_AREA) 

	return imgclip.astype('int')


def rotateImg(np_img, deg = None):
	img = ndimage.rotate(np_img, deg, mode='wrap', reshape=False)
	return img 

def rotateSeg(np_img, deg = None):
	img = ndimage.rotate(np_img, deg, mode='constant', reshape=False)
	return img 


def getAttributePts(seg):
	nrow, ncol 	= np.shape(seg)
	Nlabel 		= np.max(seg.ravel())
	attributePts = []
	attributePtsRavel = []


	for i in range(Nlabel):
		attributePts.append([])

	for i in range(nrow):
		for j in range(ncol):
			v = seg[i,j]
			if v > 0:
				attributePts[v-1].append([i,j])
				attributePtsRavel.append([i,j])

	for i in range(Nlabel):
		attributePts[i] = np.array(attributePts[i])

	return np.array(attributePts), np.array(attributePtsRavel)


def move_img_to_center(img, xpmed, ypmed):
	xshift 		 = int(np.shape(img)[1]/2 - xpmed)
	yshift 		 = int(np.shape(img)[0]/2 - ypmed)
	imgroll 	 = np.roll(img, (yshift, xshift), axis = (0,1))

	return imgroll




