import cv2
import sys
sys.path.append('../src')
from cv2_tools.cv2utils import rescale, draw_detections
import numpy as np
from apriltag_tools.Detector import Detector
folder = "thunderhill/run5_tandem/photos/DJI_0009/"

filepath_1 = "image_10.png"
filepath_2 = "image_11.png"

img1 = cv2.imread(folder+filepath_1)
img1 = rescale(img1, 100)
detector = Detector(img=img1)

results = detector.detect(turn_binary=True,
                    tag_family="tag36h11")
print(len(results))
#draw_detections(detector.img, results)
print(results[0].corners)
print(results[1].corners)
corners = results[0].corners
corners = np.array(corners).reshape((-1, 2))



low_x, low_y = np.min(corners, axis=0)
high_x, high_y = np.max(corners, axis=0)
print(low_x, low_y, high_x, high_y)
search_area = img1[low_y:high_y, low_x:high_x]
print(search_area.shape)
gray= cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kps_img1, des1 = sift.detectAndCompute(gray,None)
img_w_kp=cv2.drawKeypoints(gray,kps_img1,img1)
img_w_kp = rescale(img_w_kp, 300)

cv2.imshow("img", img_w_kp)
img2 = cv2.imread(folder+filepath_2)

detector = Detector(img=img2)

results = detector.detect( turn_binary=True,
                    tag_family="tag36h11")
print(results[1].corners)
#img2 = rescale(img2, 50)
corners = np.array(results[1].corners).reshape((-1, 2))
low_x, low_y = np.min(corners, axis=0)
high_x, high_y = np.max(corners, axis=0)
print(low_x, low_y, high_x, high_y)
search_area2 = img2[low_y:high_y, low_x:high_x]

gray2 = cv2.cvtColor(search_area2, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kps_img2, des2 = sift.detectAndCompute(gray2,None)
img_w_kp2 =cv2.drawKeypoints(gray2,kps_img2,img2)
img_w_kp2 = rescale(img_w_kp2, 300)
cv2.imshow("img2", img_w_kp2)
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    print(m.distance, n.distance)
    if m.distance< 0.75*n.distance:
        good.append([m])

img3 = cv2.drawMatchesKnn(gray, kps_img1, gray2, kps_img2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img3 = rescale(img3, 300)
cv2.imwrite("data/sift/matches_cropped.png", img3)
cv2.imshow("Matches", img3)
cv2.waitKey(0)










