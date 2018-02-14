import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelextrema
from collections import deque
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

m_per_pix_y = 30 / 720
m_per_pix_x = 3.7 / 700

def get_n_dist_maxs(array, dist, n):
	avg_val = sum(array)/len(array)
	thresholded_array = np.copy(array)
	thresholded_array[thresholded_array < avg_val*0.4] = 0

	idxs = argrelextrema(thresholded_array, np.greater_equal, order=5)[0]
	values = thresholded_array[idxs]
	
	max_list = zip(idxs, values)
	max_list = sorted(max_list, key=lambda x:x[1], reverse=True)

	if max_list:
		maxs = [max_list[0]]
		if len(max_list) > 1:
			for m in max_list[1:]:
				distance = abs(maxs[0][0] - m[0])
				if distance > dist:
					maxs.append(m)
		return np.flip(sorted(maxs[:n], key=lambda x:x[0]), axis=0).astype(int)

def calc_curv_rad(fit_coords, y_val):
	y = fit_coords[:, 1]
	x = fit_coords[:, 0]
	fit = np.polyfit(y * m_per_pix_y, x * m_per_pix_x, 2)
	return ((1 + (2*fit[0]*y_val*m_per_pix_y + fit[1])**2)**1.5) / np.absolute(2*fit[0])

def calc_lane_width(fit_left_coords, fit_right_coords, img_width, img_height):
	x_left = fit_left_coords[img_height-1][0]
	x_right = fit_right_coords[img_height-1][0]
	return np.absolute((x_right - x_left) * m_per_pix_x)

def calc_dist_to_line(fit_coords, img_width, img_height):
	x = fit_coords[img_height-1][0]
	return np.absolute((img_width//2 - x) * m_per_pix_x)

class SlidingWindowScanner:

	def __init__(self, window_num=9, window_width=100):
		self.window_num = window_num
		self.window_width = window_width
		self.windows_initialized = False
		self.minpix = 50
		self.avg_num = 5
		self.left_coeffs = deque(maxlen=self.avg_num)
		self.right_coeffs = deque(maxlen=self.avg_num)

	def get_averaged_fit(self, lane='left'):
		if lane == 'left':
			return sum(self.left_coeffs)/len(self.left_coeffs)
		if lane == 'right':
			return sum(self.right_coeffs)/len(self.right_coeffs)

	def get_curve_coords(self, img_height, coeffs):
		ploty = np.linspace(0, img_height-1, img_height)
		return np.column_stack((coeffs[0]*ploty**2 + coeffs[1]*ploty + coeffs[2], ploty))

	def get_window_rect(self, window_num, center_x, img_height):
		win_y_low = img_height - (window_num+1)*self.window_height
		win_y_high = img_height - window_num*self.window_height
		win_x_low = center_x - self.window_width
		win_x_high = center_x + self.window_width
		return [(win_x_low,win_y_low), (win_x_high,win_y_high)]

	def draw_curve(self, img, coords):
		for i in range(len(coords)-1):
			x = int(coords[i][0])
			y = int(coords[i][1])
			next_x = int(coords[i+1][0])
			next_y = int(coords[i+1][1])
			cv2.line(img, (x, y), (next_x, next_y), (255, 255, 0), 10)

	def draw_lane(self, img, left_coords, right_coords):
		int_points_l = np.array([[int(i[0]), int(i[1])] for i in left_coords])
		int_points_r = np.array([[int(i[0]), int(i[1])] for i in right_coords])
		conc = np.concatenate((int_points_l, np.flip(int_points_r, axis=0)))
		cv2.fillPoly(img, [conc], (0,255,0))

	def draw_line_area(self, img, line_fit_coords):
		left_border_coords = np.array(line_fit_coords-[self.window_width,0]).astype(int)
		right_border_coords = np.array(line_fit_coords+[self.window_width,0]).astype(int)
		line_pts = np.concatenate((left_border_coords, np.flip(right_border_coords, axis=0)))
		cv2.fillPoly(img, [line_pts], (0,255, 0))

	def initialize_lanes(self, img):
		self.window_height = np.int(img.shape[0]/self.window_num)
		histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
		histogram = gaussian_filter(histogram, sigma=7)
		lane_peaks = get_n_dist_maxs(histogram, dist=500, n=2)
		windows_layer = np.dstack((img, img, img))*0
		lines_layer = np.dstack((img, img, img))*0
		points_layer = np.dstack((img, img, img))*0
		lane_layer = np.dstack((img, img, img))*0

		#plt.imshow(img)
		#lt.show()

		if len(lane_peaks) > 1:
			left_wind_init_idx = lane_peaks[0][0]
			right_wind_init_idx = lane_peaks[1][0]

			nonzero = img.nonzero()
			nonzeroy = np.array(nonzero[0])
			nonzerox = np.array(nonzero[1])
			leftx_current = left_wind_init_idx
			rightx_current = right_wind_init_idx

			left_lane_inds = []
			right_lane_inds = []

			for window in range(self.window_num):
				left_rect = self.get_window_rect(window, leftx_current, img.shape[0])
				right_rect = self.get_window_rect(window, rightx_current, img.shape[0])

				cv2.rectangle(windows_layer,left_rect[0],left_rect[1],(0,255,0), 2) 
				cv2.rectangle(windows_layer,right_rect[0],right_rect[1],(0,255,0), 2) 

				good_left_inds = ((nonzeroy >= left_rect[0][1]) & (nonzeroy < left_rect[1][1]) & 
					(nonzerox >= left_rect[0][0]) &  (nonzerox < left_rect[1][0])).nonzero()[0]
				good_right_inds = ((nonzeroy >= right_rect[0][1]) & (nonzeroy < right_rect[1][1]) & 
					(nonzerox >= right_rect[0][0]) &  (nonzerox < right_rect[1][0])).nonzero()[0]

				left_lane_inds.append(good_left_inds)
				right_lane_inds.append(good_right_inds)
				if len(good_left_inds) > self.minpix:
					leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
				if len(good_right_inds) > self.minpix:  
					rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

			left_lane_inds = np.concatenate(left_lane_inds)
			right_lane_inds = np.concatenate(right_lane_inds)

			leftx = nonzerox[left_lane_inds]
			lefty = nonzeroy[left_lane_inds] 
			rightx = nonzerox[right_lane_inds]
			righty = nonzeroy[right_lane_inds] 

			left_fit = np.polyfit(lefty, leftx, 2)
			right_fit = np.polyfit(righty, rightx, 2)

			self.left_coeffs.append(left_fit)
			self.right_coeffs.append(right_fit)

			left_fit_coords = self.get_curve_coords(img.shape[0], left_fit)
			right_fit_coords = self.get_curve_coords(img.shape[0], right_fit)

			left_curv_rad = calc_curv_rad(left_fit_coords, img.shape[0]-1)
			right_curv_rad = calc_curv_rad(right_fit_coords, img.shape[0]-1)
			left_dist = calc_dist_to_line(left_fit_coords, img.shape[1], img.shape[0])
			right_dist = calc_dist_to_line(right_fit_coords, img.shape[1], img.shape[0])
			lane_width = calc_lane_width(left_fit_coords, right_fit_coords, img.shape[1], img.shape[0])

			points_layer[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
			points_layer[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

			self.draw_curve(lines_layer, left_fit_coords)
			self.draw_curve(lines_layer, right_fit_coords)

			self.draw_lane(lane_layer, left_fit_coords, right_fit_coords)

			self.windows_initialized = True

		return [windows_layer, lines_layer, 
			points_layer, lane_layer, 
			left_curv_rad, right_curv_rad,
			left_dist, right_dist, lane_width]

	def adjust_lanes(self, img):
		nonzero = img.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])

		left_fit = self.get_averaged_fit('left')
		right_fit = self.get_averaged_fit('right')
		left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
			left_fit[2] - self.window_width)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
			left_fit[1]*nonzeroy + left_fit[2] + self.window_width))) 

		right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
			right_fit[2] - self.window_width)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
			right_fit[1]*nonzeroy + right_fit[2] + self.window_width)))

		windows_layer = np.dstack((img, img, img))*0
		lines_layer = np.dstack((img, img, img))*0
		points_layer = np.dstack((img, img, img))*0
		lane_layer = np.dstack((img, img, img))*0

		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]

		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)

		self.left_coeffs.append(left_fit)
		self.right_coeffs.append(right_fit)

		left_fit = self.get_averaged_fit('left')
		right_fit = self.get_averaged_fit('right')

		left_fit_coords = self.get_curve_coords(img.shape[0], left_fit)
		right_fit_coords = self.get_curve_coords(img.shape[0], right_fit)

		left_curv_rad = calc_curv_rad(left_fit_coords, img.shape[0]-1)
		right_curv_rad = calc_curv_rad(right_fit_coords, img.shape[0]-1)
		left_dist = calc_dist_to_line(left_fit_coords, img.shape[1], img.shape[0])
		right_dist = calc_dist_to_line(right_fit_coords, img.shape[1], img.shape[0])
		lane_width = calc_lane_width(left_fit_coords, right_fit_coords, img.shape[1], img.shape[0])

		self.draw_line_area(windows_layer, left_fit_coords)
		self.draw_line_area(windows_layer, right_fit_coords)

		self.draw_curve(lines_layer, left_fit_coords)
		self.draw_curve(lines_layer, right_fit_coords)

		points_layer[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		points_layer[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

		self.draw_lane(lane_layer, left_fit_coords, right_fit_coords)

		return [windows_layer, lines_layer, 
			points_layer, lane_layer, 
			left_curv_rad, right_curv_rad,
			left_dist, right_dist, lane_width]

	def process_image(self, img):
		if not self.windows_initialized:
			debug_layers = self.initialize_lanes(img)
		else:
			debug_layers = self.adjust_lanes(img)
		return debug_layers
