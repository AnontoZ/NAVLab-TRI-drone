from cv2 import SOLVEPNP_P3P
import numpy as np 
import os
import dt_apriltags
import pickle
import sys
import cv2
sys.path.append('../src')
from apriltag_scripts.tag_detection_simple import get_video_tags, draw_tags

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

    # Get video from drone and initialize AprilTag detector
    vid_name = 'DJI_0009.MOV'
    vid_path = os.path.join(data_dir,vid_name)
    detector = dt_apriltags.Detector(families="tag16h5")

    # Load tag detection data
    tags_file = os.path.join(data_dir,'detected_tags.pickle')
    with open(tags_file, 'rb') as f:
        detected_tags = pickle.load(f)

    time_file = os.path.join(data_dir,'tags_times.pickle')
    with open(time_file, 'rb') as f:
        tag_times = pickle.load(f)

    ## Get estimated camera pose from cones
    # Cone coordinates in image frame (estimated from appearance)
    cone1_px = np.array([869, 1278])
    cone2_px = np.array([1315, 271])
    cone3_px = np.array([1747, 1392])
    cones_px = np.array([cone1_px, cone2_px, cone3_px],dtype="double")
    # cones_px = cones_px.reshape(3,2,1)

    # Cone coordinates in world frame
    cone1_pw = np.array([cone1[0,1], cone1[0,2], cone1[0,3]])
    cone2_pw = np.array([cone2[0,1], cone2[0,2], cone2[0,3]])
    cone3_pw = np.array([cone3[0,1], cone3[0,2], cone3[0,3]])
    cones_pw = np.array([cone1_pw, cone2_pw, cone3_pw], dtype=np.float32)
    # cones_pw = np.ascontiguousarray(cones_pw[:,:3]).reshape(3,1,3)
    # cones_pw = cones_pw.reshape(3,3,1)

    # Solve PnP problem 
    retval, rvecs, tvecs = cv2.solveP3P(cones_pw, cones_px, cam_mtx, cam_distort, flags=cv2.SOLVEPNP_P3P)
    rot_pw2px, _ = cv2.Rodrigues(rvecs[0])
    Tw = np.zeros((4,4))
    Tw[:3,:3] = rot_pw2px
    Tw[:3,-1] = tvecs[0].reshape(3,)
    Tw[-1,-1] = 1

    # Convert tag data to coordinates in frame
    pi_mat = np.eye(3,4)
    pw2pc = cam_mtx@pi_mat@Tw
    pc2pw = np.linalg.inv(pw2pc)

    print("here")
    # Iterate through tags and find corresponding positions
    # for tag_set in detected_tags:
        









