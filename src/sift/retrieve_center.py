import cv2
import sys
from datetime import date, datetime
# setting path
sys.path.append('../src')
import cv2_tools.cv2utils as cv2utils
from sift.SIFT_detector import get_matches_lowe_ratio
from rotations_2d.angle_between import angle_between, get_rotation_matrix
import numpy as np
from apriltag_tools.Detector import Detector
import matplotlib.pyplot as plt




class AprilTagSift:
    def __init__(self, result, feature_detector, img):
        self.feature_det = feature_detector

        corners = result.corners
        corners = np.array(corners).reshape((-1, 2))
        self.center = result.center
        self.low_x, self.low_y = np.min(corners, axis=0)# - np.array([500, 500])
        high_x, high_y = np.max(corners, axis=0)# + np.array([500, 500])
        self.search_area = img[self.low_y:high_y, self.low_x:high_x]
        self.save_descriptors()
    def save_descriptors(self):

        if len(self.search_area.shape)>2:
            gray = cv2.cvtColor(self.search_area, cv2.COLOR_BGR2GRAY)
        else:
            gray = self.search_area
        self.kps, self.des = self.feature_det.detectAndCompute(gray,None)
        print("Initial kps", len(self.kps))
        self.relatives = [np.array(kp.pt)+np.array([self.low_x, self.low_y])-self.center for kp in self.kps]
        self.img = gray





def get_center(good, angle, kps_2, tag_ref):
    centers = []
    for match in good:
        R = get_rotation_matrix(angle)

        match0 = match[0]
        rel = tag_ref.relatives[match0.queryIdx]
        vec1 = np.array(kps_2[match0.trainIdx].pt)
        rel_rot = R@rel
        c = vec1-rel_rot
        centers.append(c)

    return np.median(centers, axis=0)
def get_angle(good, kps_ref, kps_2):
    angles = []
    for i in range(0, len(good)-1, 2):
        if i >= len(good)-i-1:
            break
        match0 = good[i][0]
        match1 = good[len(good)-i-1][0]
        vec1 = np.array(kps_2[match0.trainIdx].pt)-np.array(kps_2[match1.trainIdx].pt)
        vec0 = np.array(kps_ref[match0.queryIdx].pt)-np.array(kps_ref[match1.queryIdx].pt)

        #Rotation from first image to second image
        angle = angle_between(vec0, vec1)
        angles.append(angle)
    ret =  np.median(angles)   
    return ret




def get_center_w_sift(img2, tag_ref):
    if len(img2.shape)>2:
        print(img2.shape)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else: 
        gray2 = img2
    kps_img2, des2 = tag_ref.feature_det.detectAndCompute(gray2,None)
    if des2 is None:
        print("des2 is none")

        return [], False
    if len(kps_img2)<4:
        print("len kps<2")
        return [], False
    good = get_matches_lowe_ratio(tag_ref.des, des2, 0.4, False, tag_ref.img, gray2, tag_ref.kps, kps_img2)
    if len(good)<2:
        #print("no good matches")
        #print(len(good))
        return [], False
    angle = get_angle(good, tag_ref.kps, kps_img2)
    c = get_center(good, angle, kps_2=kps_img2, tag_ref=tag_ref)
    return np.array(c), True

if __name__=='__main__':
    folder = "thunderhill/run5_tandem/photos/DJI_0009/"


    filepath_1 = "image_3.png"
    filepath_2 = "image_17.png"

    img1 = cv2utils.rescale(cv2.imread(folder+filepath_1), 100)
    #img1 = cv2utils.fit_to_screen(img1)

    img2 = cv2utils.rescale(cv2.imread(folder+filepath_2), 100)
    ### Save first keypoints

    t1 = datetime.now()
    detector = Detector(img=img1)
    results = detector.detect(increase_constrast=False, 
                        adaptive_threshold=False, turn_binary=True,
                        tag_family="tag36h11")
    t2 = datetime.now()
    detector = "sift"
    if detector == "brisk":
        feature_det = cv2.BRISK_create()
    if detector == "akaze":
        feature_det = cv2.AKAZE_create()
    if detector == "kaze":
        feature_det = cv2.KAZE_create()
    if detector=="orb": 
        # find the keypoints with ORB
        feature_det = cv2.ORB_create()
    else:
        feature_det = cv2.SIFT_create()


    tag1Sift = AprilTagSift(results[0], feature_detector=feature_det, img=img1)
    tag2Sift = AprilTagSift(results[1], feature_detector=feature_det, img=img1)

    print(results[0].center)
    print(results[1].center)
    c1 = get_center_w_sift(img2, tag1Sift)
    c2 = get_center_w_sift(img2, tag2Sift)
    detector = Detector(img=img2)
    results = detector.detect(increase_constrast=False, 
                        adaptive_threshold=False, turn_binary=True,
                        tag_family="tag36h11")
    print("Calculated centers", c1, c2)
    print("img2 centers", results[0].center, results[1].center)

    #print(len(good))


