import numpy as np 
import csv
import os
import dt_apriltags
import pickle
import sys
import cv2
import matplotlib.pyplot as plt 
import pandas as pd
import pymap3d as pm
sys.path.append('../src')
from apriltag_scripts.tag_detection_simple import get_video_tags, draw_tags, get_ros_blobs
from cv2_tools.CoordinateTransform import pixelToWorld
from thunderhill_analysis.color_test import detect_from_frame

if __name__=='__main__':
    '''
    Description: Script to analyze performance of vehicle detection in Thunderhill
    '''
    # Get image data from Pandas dataframes
    bag_dir = '../data/thunderhill/11-16-2022/2022-11-16-14-44-48/'
    times_fname = os.path.join(bag_dir, 'df_times.pickle')
    times_df = pd.read_pickle(times_fname)
    imgs_fname = os.path.join(bag_dir, 'df_img.pickle')
    imgs_df = pd.read_pickle(imgs_fname)

    # Get position data from csv
    start_utc = '22:44:28'
    num_values = times_df['OXTS iTOW'].size
    leader_oxts = pd.read_csv(os.path.join(bag_dir, 'leader_oxts.csv'))
    follower_oxts = pd.read_csv(os.path.join(bag_dir, 'follower_oxts.csv'))
    leader_times = leader_oxts['Time (HH:mm:ss)']
    follower_times = follower_oxts['Time (HH:mm:ss)']
    leader_start_idx = leader_times[leader_times == start_utc].index[0]
    follower_start_idx = follower_times[follower_times == start_utc].index[0]

    leader_oxts = leader_oxts.iloc[leader_start_idx:(leader_start_idx + num_values)]
    follower_oxts = follower_oxts.iloc[follower_start_idx:(leader_start_idx + num_values)]
    leader_lla = leader_oxts[['Latitude (deg)', 'Longitude (deg)', 'Altitude (m)']].to_numpy()
    follower_lla = follower_oxts[['Latitude (deg)', 'Longitude (deg)', 'Altitude (m)']].to_numpy()
    # leader_lla[:,2] = 0
    # follower_lla[:,2] = 0

    times_df_full = times_df
    times_df_full['oxts utc'] = leader_oxts['Time (HH:mm:ss)'].tolist()
    # times_df_full.to_csv(os.path.join(bag_dir,'df_times.csv'))
    times_difference = times_df_full['camera rtime'].to_numpy() - times_df_full['ublox rtime'].to_numpy()

    # Initialize AprilTag detector
    detector = dt_apriltags.Detector(families="tag16h5", quad_sigma=0.8)
    detected_tags, num_detections, vid_time = get_ros_blobs(imgs_df, display_tags=False)

    # Get camera matrix and distortion
    cam_files = np.load('../data/camera_calibration/big_drone_ros/camera_matrix.npz')
    camera_data = []
    for file in cam_files.files:
        camera_data.append(cam_files[file])
    cam_ret = camera_data[0]        # RMS reprojection error of camera matrix
    cam_mtx = camera_data[1]
    cam_distort = camera_data[2]


    # Get cone data from .csv files
    cone_dir = '../data/thunderhill/11-16-2022/cone_survey/leader_serial_ublox/'
    file_prefix = 'TH_20221116_'
    list_fnames = ['CenterCone.csv','EastCone.csv','NorthCone.csv','NorthEastCone.csv','NorthWestCone.csv',
        'SouthCone.csv', 'SouthEastCone.csv', 'SouthWestCone.csv', 'WestCone.csv']
    # cone_dir = '../data/thunderhill/11-15-2022/cone_survey/'
    # file_prefix = ''
    # list_fnames = ['CenterCone_lla.csv','EastCone_lla.csv','NorthCone_lla.csv','NorthEastCone_lla.csv','NorthWestCone_lla.csv',
    #     'SouthCone_lla.csv', 'SouthEastCone_lla.csv', 'SouthWestCone_lla.csv', 'WestCone_lla.csv']

    cones_lla = np.zeros((len(list_fnames), 3))

    for idx, cone_name in enumerate(list_fnames):
        file_name = file_prefix + cone_name
        cone_file = os.path.join(cone_dir, file_name)
        with open(cone_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            row = next(reader)
            cones_lla[idx,:] = np.array([float(row['lat']), float(row['lon']), float(row['alt [mm]'])*(10**-3)])
            # cones_lla[idx,:] = np.array([float(row['lat.']), float(row['long.']), float(row['alt.'])*(10**-3)])
            # cones_lla[idx,:] = np.array([float(row['lat']), float(row['lon']), 0])

    # Convert coordinates coordinates to ENU (relative to center cone)
    cones_lla[:,2] = 97     # NOTE: Need to normalize otherwise transformation doesn't work correctly 
    leader_lla[:,2] = 97
    follower_lla[:,2] = 97

    center_cone_lla = cones_lla[0,:]
    cones_enu = pm.geodetic2enu(cones_lla[:,0], cones_lla[:,1], cones_lla[:,2], center_cone_lla[0], center_cone_lla[1], 
        center_cone_lla[2])
    cones_enu = np.asarray(cones_enu).T
    leader_enu = pm.geodetic2enu(leader_lla[:,0], leader_lla[:,1], leader_lla[:,2], center_cone_lla[0], center_cone_lla[1], 
        center_cone_lla[2])
    leader_enu = np.asarray(leader_enu).T
    follower_enu = pm.geodetic2enu(follower_lla[:,0], follower_lla[:,1], follower_lla[:,2], center_cone_lla[0], center_cone_lla[1], 
        center_cone_lla[2])
    follower_enu = np.asarray(follower_enu).T

    # Detect cones in first frame 
    # cv2.imshow('cones', imgs_df['imgs'][1460])
    # cv2.waitKey(0)
    center_cone = (355, 244)
    east_cone = (485, 240)
    north_cone = (358, 64)
    north_east_cone = (450, 119)
    north_west_cone = (273, 104)
    south_cone = (351, 415)
    south_east_cone = (452,358)
    south_west_cone = (265,370)
    west_cone = (227, 243)
    cones_px = [center_cone, east_cone, north_cone, north_east_cone, north_west_cone, south_cone, south_east_cone, 
        south_west_cone, west_cone]
    cone_px = np.asarray(cones_px, dtype="double")

    # Solve PnP problem 
    retval, rvecs, tvecs = cv2.solvePnP(cones_enu, cone_px, cam_mtx, cam_distort)
    rot_pw2px, _ = cv2.Rodrigues(rvecs[:,0])
    Tw = np.zeros((3,4))
    Tw[:3,:3] = rot_pw2px
    Tw[:3,-1] = tvecs[:,0]              # NOTE: Here is the transformation matrix

    # Find conversion between tag coordinates in image frame to coordinates in world frame 
    # leader_pf = np.zeros((2, len(detected_tags)))              # Leader pixel coordinates 
    # follower_pf = np.copy(leader_pf)
    leader_pf = np.array([530, 244]).reshape((2,1))
    follower_pf = np.array([332, 456]).reshape((2,1))
    west_cone_pf = np.asarray(west_cone).reshape((2,1))
    south_cone_pf = np.asarray(south_cone).reshape((2,1))
    south_east_cone_pf = np.asarray(south_east_cone).reshape((2,1))
    south_west_cone_pf = np.asarray(south_west_cone).reshape((2,1))

    # Tag detection for blobs
    # TODO: adjust so it can account for multiple particles being detected
    tags_pf = np.zeros((2, len(detected_tags)))
    for idx, tag_set in enumerate(detected_tags):
        for keypoint in tag_set:
            tags_pf[:,idx] = np.asarray(keypoint.pt)

    # for idx, tag_set in enumerate(detected_tags):
    #     for tag in tag_set:
    #         if tag.tag_id == 0:
    #             leader_pf[:,idx] = tag.center
    #             continue
    #         follower_pf[:,idx] = tag.center



    # Convert to camera frame using distance from drone to AprilTag then convert to world frame 
    leader_height = 1.12649         # Tesla
    follower_height = 1.45288       # Q50
    # drone_altitude = 36 - 5
    drone_altitude = tvecs[-1]             # Close to desired
    z_leader = drone_altitude + leader_enu[0,2]
    z_follower = drone_altitude + leader_enu[0,2]           # Distance from drone to TurtleBot AprilTag (camera frame)
    leader_wf = pixelToWorld(leader_pf, cam_mtx, rot_pw2px, tvecs[:,0].reshape(3,1), z_leader)
    follower_wf = pixelToWorld(follower_pf, cam_mtx, rot_pw2px, tvecs[:,0].reshape(3,1), z_follower)
    tags_wf = pixelToWorld(tags_pf, cam_mtx, rot_pw2px, tvecs[:,0].reshape(3,1), z_follower)
    west_cone_wf = pixelToWorld(west_cone_pf, cam_mtx, rot_pw2px, tvecs[:,0].reshape(3,1), drone_altitude)
    south_cone_wf = pixelToWorld(south_cone_pf, cam_mtx, rot_pw2px, tvecs[:,0].reshape(3,1), drone_altitude)
    south_east_cone_wf = pixelToWorld(south_east_cone_pf, cam_mtx, rot_pw2px, tvecs[:,0].reshape(3,1), drone_altitude)
    south_west_cone_wf = pixelToWorld(south_west_cone_pf, cam_mtx, rot_pw2px, tvecs[:,0].reshape(3,1), drone_altitude)


    # Plot coordinates of AprilTag centers and vehicle 
    # fig, ax = plt.subplots()
    # ax.plot(tag_times, leader_wf[0,:], label='Leader (AprilTag)')
    # ax.plot(tag_times, follower_wf[0,:], label='Follower (AprilTag)')
    # ax.plot(tbot1[:,0] - tbot1[0,0], tbot1[:,1], label='TurtleBot 1')
    # ax.plot(tbot2[:,0] - tbot2[0,0], tbot2[:,1], label='TurtleBot 2')
    # ax.legend()
    # ax.set_xlabel('Time (sec)')
    # ax.set_ylabel('x-position (m)')
    # ax.grid()

    # fig, ax = plt.subplots()
    # ax.plot(tag_times, leader_wf[1,:], label='Leader (AprilTag)')
    # ax.plot(tag_times, follower_wf[1,:], label='Follower (AprilTag)')
    # ax.plot(tbot1[:,0] - tbot1[0,0], tbot1[:,2], label='TurtleBot 1')
    # ax.plot(tbot2[:,0] - tbot2[0,0], tbot2[:,2], label='TurtleBot 2')
    # ax.legend()
    # ax.set_xlabel('Time (sec)')
    # ax.set_ylabel('y-position (m)')
    # ax.grid()

    # fig, ax = plt.subplots()
    # ax.plot(tag_times, leader_wf[2,:], label='Leader (AprilTag)')
    # ax.plot(tag_times, follower_wf[2,:], label='Follower (AprilTag)')
    # ax.plot(tbot1[:,0] - tbot1[0,0], tbot1[:,3], label='TurtleBot 1')
    # ax.plot(tbot2[:,0] - tbot2[0,0], tbot2[:,3], label='TurtleBot 2')
    # ax.legend()
    # ax.set_xlabel('Time (sec)')
    # ax.set_ylabel('z-position (m)')
    # ax.grid()

    imgs_list = imgs_df['imgs'].tolist()
    # display_frame = cv2.cvtColor(imgs_list[1], cv2.COLOR_BGR2RGB)
    # plt.imshow(display_frame)

    # Get distance from blobs to vehicles
    leader_distances = []
    follower_distances = []
    outlier_idx = []
    for idx in np.arange(tags_pf.shape[1]):
        # Skip over no entries
        pixels_blob = tags_pf[:, idx]
        if np.all(pixels_blob == 0):
            continue
        
        # Skip over time offset
        data_times = times_df.iloc[idx]
        time_diff = data_times['OXTS iTOW'] - data_times['ublox iTOW']
        if np.abs(time_diff) > 0.25:
            continue

        coords_blob = tags_wf[0:2, idx]
        coords_leader = leader_enu[idx, 0:2]
        coords_follower = follower_enu[idx, 0:2]

        dist= np.linalg.norm(np.array([[coords_blob - coords_leader, coords_blob - coords_follower]]), axis=2).reshape(2,)
        # Skip erroneous blobs
        # if np.min(dist) > 5:
        #     outlier_idx.append(idx)
        #     continue
        if dist[0] < dist[1]:
            leader_distances.append(dist[0])
            continue
        follower_distances.append(dist[1])

    leader_distances = np.asarray(leader_distances)
    follower_distances = np.asarray(follower_distances)

    print(f'Number of Leader Blobs: {leader_distances.shape[0]}')
    print(f'Average of Leader Distance: {np.mean(leader_distances)}')
    print(f'Leader Distance Variance: {np.var(leader_distances)}')

    print(f'\nNumber of Follower Blobs: {follower_distances.shape[0]}')
    print(f'Average of Follower Distance: {np.mean(follower_distances)}')
    print(f'Follower Distance Variance: {np.var(follower_distances)}')

    for idx, img in enumerate(imgs_list):
        # if idx not in outlier_idx:
        #     continue
        data_times = times_df.iloc[idx]
        time_diff = data_times['OXTS iTOW'] - data_times['ublox iTOW']
        print(f'Time Difference: {time_diff}')
        pixels_blob = tags_pf[:,idx]
        if np.all(pixels_blob == 0):
            continue
        cv2.imwrite('thunderhill_ros.png', imgs_list[idx])
        display_frame = cv2.cvtColor(imgs_list[idx], cv2.COLOR_BGR2RGB)
        plt.imshow(display_frame)
        
        fig, ax = plt.subplots()
        ax.scatter(leader_enu[idx,0], leader_enu[idx,1], label='Leader (oxts)', color='red', marker='*')
        ax.scatter(follower_enu[idx,0], follower_enu[idx,1], label='Follower (oxts)', color='blue')
        ax.scatter(tags_wf[0,idx], tags_wf[1,idx], label='Blob (AprilTag)', color='green')
        ax.scatter(cones_enu[:,0], cones_enu[:,1], label='Cones', marker="^", color='black')
        ax.axis('equal')
        ax.grid()
        ax.legend()
        ax.set_xlabel('east (m)')
        ax.set_ylabel('north (m)')
        plt.show()
        plt.close("all")


    # Frame 1
    fig, ax = plt.subplots()
    ax.scatter(leader_enu[1,0], leader_enu[1,1], label='Leader (oxts)', color='red', marker='*')
    ax.scatter(follower_enu[1,0], follower_enu[1,1], label='Follower (oxts)', color='blue')
    ax.scatter(tags_wf[0,1], tags_wf[1,1], label='Blob (AprilTag)', color='green')


    # Frame 0
    # ax.scatter(leader_wf[0,:], leader_wf[1,:], label='Leader (AprilTag)', color='red', marker='*')
    # ax.scatter(follower_wf[0,:], follower_wf[1,:], label='Follower (AprilTag)', color='red')
    # ax.scatter(leader_enu[0,0], leader_enu[0,1], label='Leader (oxts)', color='blue', marker='*')
    # ax.scatter(follower_enu[0,0], follower_enu[0,1], label='Follower (oxts)', color='blue')
    # ax.plot(leader_enu[:,0], leader_enu[:,1], label='Leader (oxts)')
    # ax.plot(follower_enu[:,0], follower_enu[:,1], label='Follower (oxts)')

    ax.scatter(cones_enu[:,0], cones_enu[:,1], label='Cones', marker="^", color='black')
    # ax.scatter(west_cone_enu[0], west_cone_enu[1], label='West Cone Ublox', color='blue')
    # ax.scatter(west_cone_wf[0], west_cone_wf[1], label='West Cone Image', color='red')
    # ax.scatter(south_cone_enu[0], south_cone_enu[1], label='South Cone Ublox', color='blue')
    # ax.scatter(south_cone_wf[0], south_cone_wf[1], label='South Cone Image', color='red')
    # ax.scatter(south_west_cone_enu[0], south_west_cone_enu[1], label='South West Cone Ublox', color='blue')
    # ax.scatter(south_west_cone_wf[0], south_west_cone_wf[1], label='South West Cone Image', color='red')
    # ax.scatter(south_east_cone_enu[0], south_east_cone_enu[1], label='South East Cone Ublox', color='blue')
    # ax.scatter(south_east_cone_wf[0], south_east_cone_wf[1], label='South East Cone Image', color='red')
    ax.legend()
    ax.set_xlabel('x-position (m)')
    ax.set_ylabel('y-position (m)')
    # ax.invert_yaxis()
    ax.axis('equal')
    ax.grid()
    plt.show()

    # fig, ax = plt.subplots()
    # ax.scatter(cone1_px[0],cone1_px[1], label='Cone 1')
    # ax.scatter(cone2_px[0],cone2_px[1], label='Cone 2')
    # ax.scatter(cone3_px[0],cone3_px[1], label='Cone 3')
    # ax.scatter(pixel_cone[0,0],pixel_cone[1,0], label='Cone 1 - transform')
    # ax.scatter(pixel_cone[0,1],pixel_cone[1,1], label='Cone 2 - transform')
    # ax.scatter(pixel_cone[0,2],pixel_cone[1,2], label='Cone 3 - transform')
    # ax.plot(leader_pf[0,:], leader_pf[1,:], label='Leader (AprilTag)')
    # ax.plot(follower_pf[0,:], follower_pf[1,:], label='Follower (AprilTag)')
    # ax.plot(pixel_tbot1[0,:], pixel_tbot1[1,:], label='TurtleBot 1')
    # ax.plot(pixel_tbot2[0,:], pixel_tbot2[1,:], label='TurtleBot 2')
    # ax.legend()
    # ax.grid()
    # ax.set_title('Pixel Frame')
    # ax.axis((0, 2688, 0, 1512))
    # ax.invert_yaxis()
    # ax.xaxis.set_label_position('top') 
    # ax.axis('equal')
    # plt.show()
        









