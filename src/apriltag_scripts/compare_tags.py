from unittest import result
import cv2
import apriltag
import numpy as np
import sys
sys.path.append('../src')
from apriltag_tools.Detector import Detector
from cv2_tools.cv2utils import draw_detections, rescale

from os import listdir
from os.path import isfile, join

if __name__=='__main__':
    folder = "../data/tag_comparison/tag16h5_comparison/"
    dir_files= [f for f in listdir(folder) if isfile(join(folder, f))]
    filepaths = [folder+file for file in dir_files]


    tag_families = ["tag16h5"]
    #scale_percents = [10,9, 8,7, 6, 5]
    scale_percents = [20,19, 18, 17, 16, 15, 14]
    gaussian_blur = True
    for filepath_png in filepaths:

        for scale_percent in scale_percents:
            
            for tag_family in tag_families:
                image = cv2.imread(filepath_png)

                
                image = rescale(image, scale_percent )
                if gaussian_blur:
                    image = cv2.blur(image,(5,5))

                detector = Detector(img=image)

                results = detector.detect(turn_binary=True,
                        tag_family=tag_family)
                image = detector.img

                print("Results", tag_family, scale_percent, len(results))
                if len(results):
                    print(results[0].corners)

                extra_text = f"{tag_family}, {scale_percent}"
                draw_detections(image, results, extra_text)

