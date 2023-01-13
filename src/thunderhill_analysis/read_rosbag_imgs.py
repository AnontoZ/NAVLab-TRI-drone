import os
from re import U
import rosbag
from bagpy import bagreader
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from cv_bridge import CvBridge

if __name__ == '__main__':
    '''
    Description: Script to get data from thunderhill tests
    '''
    # Get filenames, partition into test set and verification set 
    # bag_filenames = ['2022-11-15-17-01-22.bag']
    # bag_filenames = ['2022-11-16-14-44-48.bag', '2022-11-16-14-50-25.bag', '2022-11-16-14-54-13.bag', 
    #     '2022-11-16-14-57-28.bag', '2022-11-16-14-59-58.bag', '2022-11-16-15-02-59.bag']
    # bag_folder_name = '../data/thunderhill/11-16-2022'
    # selected_file = bag_filenames[0]
    bag_folder_name = '../data/camera_calibration/big_drone_ros/'
    selected_file = '2022-12-02-14-34-29.bag'
    bag_filename = os.path.join(bag_folder_name,selected_file)

    # Open rosbag and save images to dataframe and pickle
    bag = rosbag.Bag(bag_filename)
    times = []
    imgs = []

    bridge = CvBridge()
    freq = 100
    idx = 0
    for topic, msg, t in bag.read_messages(topics='/image_view/output'):    
        if idx == 0:
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            time = t.to_time()
            times.append(time)
            imgs.append(frame)
        idx = idx + 1
        if idx > freq:
            idx = 0

    df_img = pd.DataFrame({'time': times, 'imgs': imgs})
    # df_img.to_pickle(os.path.join(bag_folder_name,selected_file[:-4],'df_img.pickle'))
    df_img.to_pickle(os.path.join(bag_folder_name,'df_img.pickle'))


