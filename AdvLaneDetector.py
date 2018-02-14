import numpy as np
import cv2
import os
import yaml
from ImagePreprocessor import ImagePreprocessor, perspective_transform
from SlidingWindowScanner import SlidingWindowScanner
from moviepy.editor import VideoFileClip
from IPython.display import HTML

class CameraParams:
	
	def __init__(self, ret=[], mtx=[], dist=[], rvecs=[], tvecs=[]):
		self.ret = ret
		self.mtx = mtx
		self.dist = dist
		self.rvecs = rvecs
		self.tvecs = tvecs 
		self.is_calibrated = False

def calc_reproj_error(obj_points, img_points, rvecs, tvecs, mtx, dist):
	mean_error = 0
	for i in range(len(obj_points)):
		img_points2, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
		error = cv2.norm(img_points[i],img_points2, cv2.NORM_L2)/len(img_points2)
		mean_error += error
	return mean_error/len(obj_points)

class AdvLaneDetector:

	def __init__(self):
		self.camera = CameraParams()
		self.img_preproc = ImagePreprocessor()
		self.sw_scanner = SlidingWindowScanner()

	def write_camera_params(self, file_name):
		data = {
			'rms': self.camera.ret,
			'camera_matrix': self.camera.mtx.tolist(),
			'dist': self.camera.dist.tolist()
		}
		with open(file_name, 'w') as file:
			yaml.dump(data, file)

	def read_camera_params(self, file_name):
		with open(file_name, 'r') as file:
			loaded_data = yaml.load(file)
			self.camera.ret = loaded_data['rms']
			self.camera.dist = np.array(loaded_data['dist'])
			self.camera.mtx = np.array(loaded_data['camera_matrix'])
			self.camera.is_calibrated = True

	def calibrate_camera(self, img_set, nx, ny):
		obj_points = []
		img_points = []

		objp = np.zeros((nx*ny, 3), np.float32)
		objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

		for img in img_set:
			gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			found, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
			if found:
				img_points.append(corners)
				obj_points.append(objp)
			else:
				print('WARNING: corners not found!')

		if img_points:
			self.camera.ret, self.camera.mtx, self.camera.dist, self.camera.rvecs, self.camera.tvecs = \
				cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
			reproj_err = calc_reproj_error(obj_points, img_points, \
				self.camera.rvecs, self.camera.tvecs, self.camera.mtx, self.camera.dist)

			print('Camera calibration reprojection error: ' + str(reproj_err))
			
		else:
			print('ERROR: no img points found!')
			self.camera.is_calibrated = False
			return

		self.camera.is_calibrated = True

	def process_image(self, img):
		if not self.camera.is_calibrated:
			print('ERROR: camera is not calibrated!')
			print('Calibrate or load calibration parameters first')
			return img

		undist = cv2.undistort(img, self.camera.mtx, self.camera.dist, None, self.camera.mtx)

		preproc_img = self.img_preproc.process_image(undist)

		[windows_layer, lines_layer, 
			points_layer, lane_layer, 
			left_rad, right_rad,
			left_dist, right_dist, lane_width] = self.sw_scanner.process_image(preproc_img)

		data_overlay = img.copy()
		cv2.rectangle(data_overlay, (0,0), (img.shape[1], img.shape[0]//4), (0,0,0), -1)

		lines_layer_transformed = perspective_transform(lines_layer, reverse=True)
		lane_layer_transformed = perspective_transform(lane_layer, reverse=True)
		img_bird_view = perspective_transform(img)

		result = cv2.addWeighted(img, 0.6, data_overlay, 0.4, 0.)
		result = cv2.addWeighted(result, 1., lane_layer_transformed, .5, 0.)
		result = cv2.addWeighted(result, 1., lines_layer_transformed, 1., 0.)

		scanning_window = cv2.addWeighted(img_bird_view, 1., points_layer, .5, 0.)
		scanning_window = cv2.addWeighted(scanning_window, 1., windows_layer, .1, 0.)
		scanning_window = cv2.addWeighted(scanning_window, 1., points_layer, 1., 0.)
		scanning_window = cv2.addWeighted(scanning_window, 1., lines_layer, 1., 0.)

		small_scanning_window = cv2.resize(scanning_window, (0,0), fx=0.25, fy=0.25)

		x_offset = img.shape[1] - small_scanning_window.shape[1]
		y_offset = 0
		result[y_offset:y_offset + small_scanning_window.shape[0], 
			x_offset:x_offset + small_scanning_window.shape[1]] = small_scanning_window

		cv2.putText(result, 'Radius of curvature: {:.2f} m'.format((left_rad+right_rad)/2), 
			(img.shape[1]//10, img.shape[0]//10), 
			cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

		lane_center_x = lane_width/2
		dist_from_center_1 = abs(lane_center_x - left_dist)
		dist_from_center_2 = abs(lane_center_x - right_dist)
		avg_dist_from_center = (dist_from_center_1 + dist_from_center_2)/2
		dist_text = 'Distance from center: {:.2f} m to the '.format(avg_dist_from_center)
		if left_dist > right_dist:
			dist_text += 'right'
		else:
			dist_text += 'left'

		cv2.putText(result, dist_text, 
			(img.shape[1]//10, img.shape[0]//10 + 30), 
			cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
		cv2.putText(result, 'Left distance: {:.2f} m'.format(left_dist), 
			(img.shape[1]//10, img.shape[0]//10 + 60), 
			cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
		cv2.putText(result, 'Right distance: {:.2f} m'.format(right_dist), 
			(img.shape[1]//10, img.shape[0]//10 + 90), 
			cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
		#cv2.putText(result, 'Lane width: {:.2f} m'.format(lane_width), 
		#	(img.shape[1]//10, img.shape[0]//10 + 90), 
		#	cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

		return result

#img_set = []
ld = AdvLaneDetector()

#path = './camera_cal/'
#for file_name in os.listdir(path):
#	if file_name.endswith('.jpg'):
#		print('Adding image: ' + file_name)
#		img = cv2.imread(path + file_name)
#		img_set.append(img)

#ld.calibrate_camera(img_set, nx=9, ny=6)
ld.read_camera_params('./camera_params.yml')

#detector = SimpleLaneDetector()

def process_image(image):
    res = ld.process_image(image)

    return res

output = './project_video_annotated1.mp4'
clip1 = VideoFileClip("./project_video.mp4")#.subclip(5,6)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(output, audio=False)


