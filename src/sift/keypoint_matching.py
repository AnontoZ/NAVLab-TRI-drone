import cv2
import sys
sys.path.append('../src')
from matplotlib.pyplot import get
from cv2_tools.cv2utils import rescale, draw_detections
from sift.SIFT_detector import get_matches_lowe_ratio
import numpy as np
from apriltag_tools.Detector import Detector
folder = "thunderhill/run5_tandem/photos/DJI_0009/"
sift = cv2.SIFT_create()

filepath_1 = "image_10.png"
filepath_2 = "image_17.png"

img1 = rescale(cv2.imread(folder+filepath_1), 30)
img2 = rescale(cv2.imread(folder+filepath_2), 30)



gray= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
kps_img1, des1 = sift.detectAndCompute(gray,None)



gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kps_img2, des2 = sift.detectAndCompute(gray2,None)
good = get_matches_lowe_ratio(des1, des2, 0.75, True, gray, gray2, kps_img1, kps_img2)









