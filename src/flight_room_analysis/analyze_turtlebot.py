from cv2 import COLORMAP_CIVIDIS
import numpy as np 
import os
import dt_apriltags
import pickle
import sys
import cv2
import matplotlib.pyplot as plt 
import matplotlib as mpl
sys.path.append('../src')
from apriltag_scripts.tag_detection_simple import get_video_tags, draw_tags
from cv2_tools.CoordinateTransform import pixelToWorld

if __name__=='__main__':
    '''
    Description: Script to analyze performance of AprilTag detection in flight room test
    '''

    # Get data from drone test .csv files
    data_dir = '../data/flight_room/2022-10-26-15-06-35/'
    list_fnames = ['cone_01.csv','cone_02.csv','cone_03.csv','drone.csv','turtlebot_01.csv','turtlebot_02.csv']
    data = []
    for file in list_fnames:
        data.append(np.genfromtxt(os.path.join(data_dir,file)))
    cone1 = data[0]
    cone2 = data[1]
    cone3 = data[2]
    drone = data[3]
    tbot1 = data[4]
    tbot2 = data[5]

    # Get camera matrix and distortion
    cam_files = np.load('../data/camera_calibration/mavic2/camera_matrix.npz')
    camera_data = []
    for file in cam_files.files:
        camera_data.append(cam_files[file])
    cam_ret = camera_data[0]        # RMS reprojection error of camera matrix
    cam_mtx = camera_data[1]
    cam_distort = camera_data[2]

    # Load tag detection data
    tags_file = os.path.join(data_dir,'detected_tags.pickle')
    with open(tags_file, 'rb') as f:
        detected_tags = pickle.load(f)

    time_file = os.path.join(data_dir,'tags_times.pickle')
    with open(time_file, 'rb') as f:
        tag_times = pickle.load(f)

    ## Get estimated camera pose from cones
    # Cone coordinates in image frame (estimated from appearance)
    cone3_px = np.array([869, 1278])
    cone2_px = np.array([1315, 271])
    cone1_px = np.array([1747, 1392])
    cones_px = np.array([cone1_px, cone2_px, cone3_px],dtype="double")

    # Cone coordinates in world frame
    cone1_pw = np.array([cone1[0,1], cone1[0,2], cone1[0,3]])
    cone2_pw = np.array([cone2[0,1], cone2[0,2], cone2[0,3]])
    cone3_pw = np.array([cone3[0,1], cone3[0,2], cone3[0,3]])
    cones_pw = np.array([cone1_pw, cone2_pw, cone3_pw], dtype=np.float32)

    # Solve PnP problem 
    retval, rvecs, tvecs = cv2.solveP3P(cones_pw, cones_px, cam_mtx, cam_distort, flags=cv2.SOLVEPNP_P3P)
    rot_pw2px, _ = cv2.Rodrigues(rvecs[0])
    Tw = np.zeros((3,4))
    Tw[:3,:3] = rot_pw2px
    Tw[:3,-1] = tvecs[0].reshape(3,)

    # Find conversion between tag coordinates in image frame to coordinates in world frame 
    leader_pf = np.zeros((2, len(detected_tags)))              # Leader pixel coordinates 
    follower_pf = np.copy(leader_pf)

    for idx, tag_set in enumerate(detected_tags):
        for tag in tag_set:
            if tag.tag_id == 0:
                leader_pf[:,idx] = tag.center
                continue
            follower_pf[:,idx] = tag.center

    # Convert to camera frame using distance from drone to AprilTag then convert to world frame 
    z_cf = drone[0,3] - tbot1[0,3]           # Distance from drone to TurtleBot AprilTag (camera frame)
    leader_wf = pixelToWorld(leader_pf, cam_mtx, rot_pw2px, tvecs[0].reshape(3,1), z_cf)
    follower_wf = pixelToWorld(follower_pf, cam_mtx, rot_pw2px, tvecs[0].reshape(3,1), z_cf)
    cones_wf = pixelToWorld(cones_px.T, cam_mtx, rot_pw2px, tvecs[0].reshape(3,1), drone[0,3])

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

    cones_pwh = np.ones((4, 3))
    cones_pwh[:3,:3] = cones_pw.T
    pixel_cone_h = cam_mtx@Tw@cones_pwh
    pixel_cone = pixel_cone_h[:2,:]/pixel_cone_h[-1,:]
    # print(pixel_cone)

    # Convert Turtlebot coordinates to pixel frame
    tbot1_pw = tbot1[:,1:4]
    tbot1_pwh = np.ones((4, tbot1_pw.shape[0]))
    tbot1_pwh[0:3,:] = tbot1_pw.T
    pixel_tbot1_h = cam_mtx@Tw@tbot1_pwh
    pixel_tbot1 = pixel_tbot1_h[:2,:]/pixel_tbot1_h[-1,:]

    tbot2_pw = tbot2[:,1:4]
    tbot2_pwh = np.ones((4, tbot2_pw.shape[0]))
    tbot2_pwh[0:3,:] = tbot2_pw.T
    pixel_tbot2_h = cam_mtx@Tw@tbot2_pwh
    pixel_tbot2 = pixel_tbot2_h[:2,:]/pixel_tbot2_h[-1,:]

    fig, ax = plt.subplots()
    cm = plt.get_cmap('viridis')
    iter = 6
    ax.set_color_cycle([cm(1.*i/(iter+1)) for i in range(1,iter+2)])
    # ax.plot(leader_wf[0,:], leader_wf[1,:], label='Leader (AprilTag)')
    # ax.plot(follower_wf[0,:], follower_wf[1,:], label='Follower (AprilTag)')
    # ax.plot(tbot1[:,1], tbot1[:,2], label='TurtleBot 1')
    # ax.plot(tbot2[:,1], tbot2[:,2], label='TurtleBot 2')
    ax.scatter(cone1[0,1], cone1[0,2], label='Cone 1 (optitrack)')
    ax.scatter(cone2[0,1], cone2[0,2], label='Cone 2 (optitrack)')
    ax.scatter(cone3[0,1], cone3[0,2], label='Cone 3 (optitrack)')
    ax.scatter(cones_wf[0,0], cones_wf[1,0], label='Cone 1 (drone)')
    ax.scatter(cones_wf[0,1], cones_wf[1,1], label='Cone 2 (drone)')
    ax.scatter(cones_wf[0,2], cones_wf[1,2], label='Cone 3 (drone)') 
    ax.legend()
    ax.set_xlabel('x-position (m)')
    ax.set_ylabel('y-position (m)')
    ax.axis((-1.5, 1, -.5, 0.9))
    ax.axis('equal')
    ax.grid()

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
    plt.show()
        









