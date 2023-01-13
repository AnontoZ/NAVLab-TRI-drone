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
    bag_filenames = ['2022-11-16-14-44-48.bag', '2022-11-16-14-50-25.bag', '2022-11-16-14-54-13.bag', 
        '2022-11-16-14-57-28.bag', '2022-11-16-14-59-58.bag', '2022-11-16-15-02-59.bag']
    bag_folder_name = '../data/thunderhill/11-16-2022'
    selected_file = bag_filenames[0]
    bag_filename = os.path.join(bag_folder_name,selected_file)

    # Open rosbag and save data as to dataframe and pickle
    b = bagreader(bag_filename)
    bag = rosbag.Bag(bag_filename)

    CAM_MSG = b.message_by_topic('/image_view/output')
    df_cam = pd.read_csv(CAM_MSG)
    df_cam.to_pickle(os.path.join(bag_folder_name,selected_file[:-4],'df_cam_raw.pickle'))

    UBLOX_MSG = b.message_by_topic('/ublox_gps/navsol')
    df_ublox = pd.read_csv(UBLOX_MSG)    
    df_ublox.head()
    df_ublox.to_pickle(os.path.join(bag_folder_name,selected_file[:-4],'df_ublox_raw.pickle'))


