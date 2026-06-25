# ::::::::::::::::PYTHON CODE::::::::::::::::
# http://ConsultingJoe.com

import cv2
import numpy as np
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

width = 640
height = 640

#Define the cameras to use (camera_1, camera_2)
camera_1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera_1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera_1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
camera_1.set(cv2.CAP_PROP_FPS, 15)

camera_2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
camera_2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera_2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
camera_2.set(cv2.CAP_PROP_FPS, 15)

img_left = cv2.imread('1.jpg')
img_right = cv2.imread('r1.jpg')

segmentor = SelfiSegmentation(0)

# Create a stereo block matching object
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=11)
stereo = cv2.StereoSGBM_create(
    minDisparity=12,
    numDisparities=64,
    blockSize=21)

while True:
    camera_1.grab()
    camera_2.grab()

    Read the frames from each camera
    ret1, frame1 = camera_1.read()
    ret2, frame2 = camera_2.read()

    frame1 = segmentor.removeBG(frame1, (0,0,0))
    frame2 = segmentor.removeBG(frame2, (0,0,0))

    # Convert the frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((3,3), np.float32)/9
    gray1 = cv2.filter2D(gray1, -1, kernel)
    gray2 = cv2.filter2D(gray2, -1, kernel)

    # Compute the depth map using the stereo block matching object
    depth_map = stereo.compute(gray1, gray2)
    # Normalize the depth map to be between 0 and 255
    norm_depth = cv2.normalize(depth_map, None, 10, 245, cv2.NORM_MINMAX)

    # Convert the normalized depth map to color
    color_depth = cv2.applyColorMap(norm_depth.astype(np.uint8), cv2.COLORMAP_HSV)
    color_depth = cv2.cvtColor(color_depth, cv2.COLOR_BGR2GRAY)

    # Stack the original frames vertically, along with the color depth map
    result = np.hstack((frame1, frame2, norm_depth))
    color_depth = color_depth[color_depth.shape[0]//2:color_depth.shape[0]]
    result = result[:, :-300]
    color_depth = cv2.flip(color_depth, 1)
    color_depth = color_depth[:height, :width-100]

    # Display the stacked frames and depth map
    cv2.imshow("3D Video", color_depth)
    cv2.imshow("Stereo Video and Depth Map", cv2.flip(result, 1))
    cv2.waitKey(10000)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the cameras
camera_1.release()
camera_2.release()