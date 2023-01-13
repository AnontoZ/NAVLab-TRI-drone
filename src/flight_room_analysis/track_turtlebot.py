from tkinter import Frame
import numpy as np 
import os
import dt_apriltags
import pickle
import sys
import cv2
import matplotlib.pyplot as plt 
sys.path.append('../src')
from apriltag_scripts.tag_detection_simple import get_video_tags, draw_tags
from cv2_tools.CoordinateTransform import pixelToWorld

if __name__=='__main__':
    '''
    Description: Script to track AprilTag locations in flight room test
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
    t = tvecs[0].reshape((3,1))

    # Initialize trackers
    trackerLeader = cv2.TrackerCSRT_create()
    trackerFollower = cv2.TrackerCSRT_create()

    # Read video and set data structures
    vid_name = "DJI_0009.MP4"
    filepath_vid = os.path.join(data_dir, vid_name)
    cap = cv2.VideoCapture(filepath_vid)
    is_read, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get first tags and put bounding box on AprilTags
    tag_set0 = detected_tags[0]
    for tag in tag_set0:
        corners = tag.corners.astype(np.int32)
        x_corners = corners[:,0]
        y_corners = corners[:,1]
        bbox = (np.min(x_corners), np.min(y_corners), np.max(x_corners), np.max(y_corners))
        if tag.tag_id == 0:
            trackerLeader.init(frame, bbox)
            continue
        trackerFollower.init(frame, bbox)

    cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking", 900, 600)

    display_track = True
    while True:
        is_read, frame = cap.read()
        # Break loop if no more frames remain
        if not is_read:
            break
    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect tags, store number of detections, tag IDs and associated time
        retLeader, bboxLeader = trackerLeader.update(frame)
        
        # Display image with tracking
        if display_track:
            p1 = (int(bboxLeader[0]), int(bboxLeader[1]))
            p2 = (int(bboxLeader[0] + bboxLeader[2]), int(bboxLeader[1] + bboxLeader[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        cv2.imshow("Tracking", frame)
        cv2.waitKey(1)

    cap.release()    
    cv2.destroyAllWindows()

    print("Got here")