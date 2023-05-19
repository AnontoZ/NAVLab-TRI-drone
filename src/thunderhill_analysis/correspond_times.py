import os
import pandas as pd
import numpy as np
from cv_bridge import CvBridge
import cv2

if __name__=='__main__':
    '''
    Description: Script to find the closest pictures to a corresponding oxts iTOW time
    '''
    # Import dataframes
    bag_folder_name = '../data/thunderhill/11-16-2022/2022-11-16-14-44-48/'
    cam_fname = os.path.join(bag_folder_name, 'df_cam_raw.pickle')
    cam_df = pd.read_pickle(cam_fname)

    ublox_fname = os.path.join(bag_folder_name, 'df_ublox_raw.pickle')
    ublox_df = pd.read_pickle(ublox_fname)

    imgs_fname = os.path.join(bag_folder_name, 'df_img_raw.pickle')
    imgs_df = pd.read_pickle(imgs_fname)

    # Start and end times (sec)
    start_time = 341086
    end_time = 341166
    oxts_iTOW = np.arange(start_time, end_time+1)

    # ubox iTOW time in seconds
    ublox_iTOW = ublox_df['iTOW']*(10**-3) + ublox_df['fTOW']*(10**-9)
    ublox_iTOW = ublox_iTOW.to_numpy()     # NOTE: Leap seconds                 

    # find closest ublox time to each oxts time 
    ublox_close_idx = np.argmin(np.abs(np.repeat(oxts_iTOW.reshape(-1,1), ublox_iTOW.shape[0], axis=1) - 
        np.repeat(ublox_iTOW.reshape(1,-1), oxts_iTOW.shape[0], axis=0)), axis=1)
    ublox_df = ublox_df.iloc[ublox_close_idx]

    ublox_time = ublox_df['Time'].to_numpy()
    cam_time = cam_df['Time'].to_numpy()

    # Find closest camera time to each ublox time 
    cam_close_idx = np.argmin(np.abs(np.repeat(ublox_time.reshape(-1,1), cam_time.shape[0], axis=1) - 
        np.repeat(cam_time.reshape(1,-1), ublox_time.shape[0], axis=0)), axis=1)

    cam_df = cam_df.iloc[cam_close_idx]

    df_times = pd.DataFrame({'OXTS iTOW': oxts_iTOW, 'ublox iTOW': (10**-3)*ublox_df['iTOW'].to_numpy(),
        'ublox rtime': ublox_df['Time'].to_numpy(), 'camera rtime': cam_df['Time'].to_numpy()})


    imgs_df = imgs_df.iloc[cam_close_idx]

    # for img in imgs_df['imgs']:
    cv2.imshow("First image", imgs_df['imgs'].iloc[0])
    cv2.waitKey(0)
    check_equal = np.allclose(imgs_df['time'].to_numpy(), cam_df['Time'].to_numpy(), 1e-5)
    assert check_equal

    # Save new dataframes with matching
    df_times.to_pickle(os.path.join(bag_folder_name,'df_times.pickle'))
    cam_df.to_pickle(os.path.join(bag_folder_name,'df_cam.pickle'))
    imgs_df.to_pickle(os.path.join(bag_folder_name, 'df_img.pickle'))

    df_times.to_csv(os.path.join(bag_folder_name,'df_times.csv'))
    
    

