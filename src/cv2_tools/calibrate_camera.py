import numpy as np
import cv2 as cv
import glob
import os
import pickle
import pandas as pd

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*9,3), np.float32)
# Squares are 20mm-by-20mm
objp[:,:2] = 20*np.mgrid[0:7,0:9].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
folder_name = '../data/camera_calibration/big_drone_ros/'
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
# images = glob.glob(os.path.join(folder_name,'*.png'))

bag_dir = '../data/camera_calibration/big_drone_ros/'
imgs_fname = os.path.join(bag_dir, 'df_img.pickle')
imgs_df = pd.read_pickle(imgs_fname)
images = imgs_df['imgs'].tolist()
img_count = 0

# for fname in images:
for img in images:
    # img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,9), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        img_count = img_count + 1
        # Draw and display the corners
        # cv.drawChessboardCorners(img, (7,6), corners2, ret)
        # cv.namedWindow("img", cv.WINDOW_NORMAL)
        # cv.resizeWindow("img", 900, 600)
        # cv.imshow('img', img)
        # cv.waitKey(0)
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print(f'Reprojection Error (want as close to zero as possible): {ret}')
print(f'Calibration Matrix: {mtx}')
print(f'Distortion Coeffs: {dist}')
print(f'Number of Images Used: {img_count}')

np.savez(os.path.join(folder_name,'camera_matrix.npz'),ret=ret,mtx=mtx,dist=dist)