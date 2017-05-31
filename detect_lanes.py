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
    out_img = lane_obj['out_img']
    return out_img

clip = VideoFileClip('project_video.mp4')#.subclip(0, 15)
newclip = clip.fl(lambda gf, t: transform(gf(t)))
newclip.write_videofile("final_video.mp4")
