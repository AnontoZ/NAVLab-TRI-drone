import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import sys

if __name__=='__main__':
    # setting path
    sys.path.append('../src')
    import cv2_tools.cv2utils as cv2utils


    folder = "thunderhill/run_4/DJI_0005/"
    dir_files= [f for f in listdir(folder) if isfile(join(folder, f))]
    filepaths = [folder+file for file in dir_files]
    for filepath in filepaths:
        img = cv2.imread(filepath)
        #cv2.imshow("Image", img)
        #cv2.waitKey(0)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        mid_red = np.array([0, 0, 180])
        #high_range = np.array([150, 255, 200])
        high_range = np.array([150, 150, 255])
        red_mask = cv2.inRange(img, lowerb = mid_red, upperb=high_range)

        red = cv2.bitwise_and(img,img, mask=red_mask)
        red = cv2utils.rescale(red, 30)

        print()
        np.nonzero(red)
        cv2.imshow("red", red)
        cv2.waitKey(0)
