import numpy as np
import pickle
import cv2
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from  moviepy.editor import VideoFileClip
from lane_functions import *

# Read in an image
#image = mpimg.imread('test_images/color-shadow-example.jpg')
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def transform(image):
    binary_warped, undist, M = thresh_warp(image, mtx, dist, ch_thresh=(170, 255), sx_thresh=(30, 100))
    lane_obj = lane_lines(binary_warped, image, undist, inv(M))
    left_fit = lane_obj['left_fit']
    right_fit = lane_obj['right_fit']
    left_curverad = lane_obj['left_curverad']
    right_curverad = lane_obj['right_curverad']
    curverad = (left_curverad + right_curverad) / 2
    drift = lane_obj['drift']
    out_img = lane_obj['out_img']
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out_img,'Radius of Curvature = ' + str(int(curverad)) + '(m)',(0,130), font, 1, (200,255,155), 2, cv2.LINE_AA)
    if drift > 0:
        pos = 'left'
    else:
        pos = 'right'
    cv2.putText(out_img,'Vehicle is ' + str.format('{0:.3f}', abs(drift)) + 'm ' + pos + ' of center',(0,230), font, 1, (200,255,155), 2, cv2.LINE_AA)
    return out_img

clip = VideoFileClip('project_video.mp4')
newclip = clip.fl(lambda gf, t: transform(gf(t)))
newclip.write_videofile("final_video.mp4")
