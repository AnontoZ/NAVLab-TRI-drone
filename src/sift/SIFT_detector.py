import numpy as np
import cv2

from cv2_tools.cv2utils import rescale, fit_to_screen

def get_kps_and_descriptors(img, draw = False):

    raise Exception
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(gray,None)
    if draw:
        img_w_kp = cv2.drawKeypoints(gray,kps,img)
        cv2.imshow("img2", img_w_kp)
    return kps, des
def get_matches_lowe_ratio(des1, des2, lowe_ratio = 0.75, draw=False, img1=None, img2=None, kps1=None, kps2 = None):
    bf = cv2.BFMatcher()
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except Exception as e:
        print(e)
    good = []
    if len(matches) <2:
        return []
    

    for m,n in matches:
        if m.distance< lowe_ratio*n.distance:
            good.append([m])

    if draw:

        img3 = cv2.drawMatchesKnn(img1, kps1, img2, kps2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        img3 = fit_to_screen(img3)
        cv2.imshow("Matches", img3)
        cv2.waitKey(0)
    return good