import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from lane_functions import *

# Read in the saved objpoints and imgpoints
nx = 9
ny = 6

# Read in an image

dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]


def undistort(img):
    img_size = (img.shape[1], img.shape[0])
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    #src = np.float32([[590, 453], [691, 453], [201, 719], [1097, 719]])
    #dst = np.float32([[250, 0], [1000, 0], [250, 720], [1000, 720]])
    #M = cv2.getPerspectiveTransform(src, dst)
    #undistorted = cv2.warpPerspective(undistorted, M, img_size)
    return undistorted

img = mpimg.imread('test_images/test2.jpg')
#img = mpimg.imread('camera_cal/calibration2.jpg')
undistorted = undistort(img)
mpimg.imsave('test_images/test1.jpg', undistorted)
#undistorted, _ = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.savefig('camera_cal/undistort_output.png')
plt.show()
