import cv2
import numpy as np
import matplotlib.pyplot as plt

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if orient == 'x':
	    abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
	if orient == 'y':
	    abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	binary_output = np.zeros_like(scaled_sobel)
	binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
	return binary_output

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	gradmag = np.sqrt(sobelx**2 + sobely**2)
	scale_factor = np.max(gradmag)/255 
	gradmag = (gradmag/scale_factor).astype(np.uint8) 
	binary_output = np.zeros_like(gradmag)
	binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
	return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
	absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
	binary_output =  np.zeros_like(absgraddir)
	binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
	return binary_output

def hls_single_chan_thresh(img, chan='s', thresh=(0,255)):
	hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	chan_list = ['h','l','s']
	chan_num = chan_list.index(chan)
	channel = hls[:,:,chan_num]
	binary = np.zeros_like(channel)
	binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
	return binary

def perspective_transform(img, reverse=False):
	src = np.float32([[558,480],[735,480],[279,669],[1048,669]])
	dst = np.float32([[305,480/4],[1022,480/4],[279,669],[1048,669]])

	if not reverse:
		M = cv2.getPerspectiveTransform(src, dst)
	else:
		M = cv2.getPerspectiveTransform(dst, src)

	img_size = (img.shape[1], img.shape[0])
	warped = cv2.warpPerspective(img, M, img_size)

	return warped

class ImagePreprocessor:

	def __init__(self, ksize=3,
		gradx_thresh=(20, 100), grady_thresh=(80, 200),
		mag_thresh=(80, 200),dir_thresh=(0.7, 1.3)):

		self.ksize = ksize
		self.gradx_thresh = gradx_thresh
		self.grady_thresh = grady_thresh
		self.mag_thresh = mag_thresh
		self.dir_thresh = dir_thresh

	def process_image(self, img):
		gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=self.ksize, thresh=self.gradx_thresh)
		grady = abs_sobel_thresh(img, orient='y', sobel_kernel=self.ksize, thresh=self.grady_thresh)
		mag_binary = mag_thresh(img, sobel_kernel=self.ksize, thresh=self.mag_thresh)
		dir_binary = dir_threshold(img, sobel_kernel=self.ksize, thresh=self.dir_thresh)

		combined_grad_thresh = np.zeros_like(dir_binary).astype(np.uint8) 
		combined_grad_thresh[((gradx == 1) & (grady == 1)) | 
							((mag_binary == 1) & (dir_binary == 1))] = 1

		color_thresh_s = hls_single_chan_thresh(img, 's', (140, 255))
		color_thresh_l = hls_single_chan_thresh(img, 'l', (80, 255))

		color_thresh = np.zeros_like(color_thresh_s)
		color_thresh[(color_thresh_s == 1) & 
						(color_thresh_l == 1)] = 1

		combined_binary = np.zeros_like(combined_grad_thresh)
		combined_binary[(color_thresh == 1) | 
						(combined_grad_thresh == 1)] = 1	

		masked_img = combined_binary# * 255

		bird_view = perspective_transform(masked_img)

		return bird_view









