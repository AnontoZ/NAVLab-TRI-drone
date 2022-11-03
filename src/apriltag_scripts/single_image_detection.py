import cv2
import apriltag
import numpy as np
import sys
sys.path.append('../src')
from apriltag_tools.Detector import Detector
import os
from cv2_tools.cv2utils import *
print(cv2.__version__)

if __name__ == '__main__':
    '''
    Description: script to detect AprilTags in a still image given an image path 
    '''
    # folder = "thunderhill/run_4/DJI_0004/"
    # image_name = "image_29"

    # Get path to image
    folder = "../thunderhill/run5_tandem/dt2e-5/"
    image_name = "image_1"
    filepath_png = folder + image_name +".png"

    # Create AprilTag detector for image
    detector = Detector(filepath_png, None)

    # Detect tags and return results
    results = detector.detect(turn_binary=True, units=8, visualize=False)
    image = detector.img

    print("Results", len(results))

    draw_detections(image, results)

