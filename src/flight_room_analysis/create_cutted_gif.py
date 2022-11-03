import rosbag
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv_bridge
import glob
import sys
sys.path.append('../src')
from cv2_tools.cv2utils import rescale

if __name__ == '__main__':
    folder = "../thunderhill/run_4/DJI_0004/"
    max_idx = 30
    img_array = []
    for i in range(1, max_idx+1):
        #print(img.shape)
        img = cv2.imread(folder+"image_"+str(i)+".png")[300:, 500:3500]
        img = rescale(img, 30)

        cv2.imshow('frame', img)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
        if cv2.waitKey(1) == ord('q'):
            break

    out = cv2.VideoWriter('data/flightroom_traj/videos/video2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
            
