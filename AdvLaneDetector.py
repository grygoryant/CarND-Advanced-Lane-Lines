from Line import Line
import numpy as np
import cv2
import os
import yaml

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
		self.left_line = Line()
		self.right_line = Line()
		self.camera = CameraParams()

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
		undist = cv2.undistort(img, self.camera.mtx, self.camera.dist, None, self.camera.mtx)

		return undist

img_set = []
ld = AdvLaneDetector()

path = './camera_cal/'
for file_name in os.listdir(path):
	if file_name.endswith('.jpg'):
		print('Adding image: ' + file_name)
		img = cv2.imread(path + file_name)
		img_set.append(img)

ld.calibrate_camera(img_set, nx=9, ny=6)
ld.write_camera_params('./camera_params.yml')
#ld.read_camera_params('./camera_params.yml')
img = cv2.imread('./camera_cal/calibration20.jpg')
ld.process_image(img)
