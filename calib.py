import numpy as np
import cv2
import glob
import os.path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from lane_functions import *


nx = 9
ny = 6
calib_folder = 'camera_cal'

# Get object and image points
objpoints, imgpoints = get_corners(calib_folder, nx, ny)

# Read in an image
img = mpimg.imread('camera_cal/calibration3.jpg')
# Get the calibration matrix and distortion
mtx, dist = cal_dist(img, objpoints, imgpoints)
dist = {'mtx': mtx, 'dist': dist}

with open('wide_dist_pickle.p', 'wb') as f:
    pickle.dump(dist, f)
