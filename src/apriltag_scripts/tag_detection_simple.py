import dt_apriltags
import cv2
import sys
import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import pickle

from zmq import Frame
sys.path.append('../src')
from cv2_tools.cv2utils import *
from apriltag_tools.Detector import Detector

def draw_tags(img, tags):
    '''
    Description: draws locations of tags on image
    Inputs:
        - img: image where tags were detected
        - tags: Array of dt_apriltags.Detection objects
    '''
    for tag in tags:
        corners = np.array(tag.corners, np.int32)
        corners = corners.reshape((-1, 1, 2))
        isClosed = True
        color = (0, 255, 0) # in (BGR)
        thickness = 20       # ptx  
        img = cv2.polylines(img, corners, isClosed, color, thickness)

    cv2.namedWindow("Detected AprilTags", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Detected AprilTags", 900, 600)
    cv2.imshow("Detected AprilTags", img)
    cv2.waitKey(0)

def get_video_tags(video_file, detector, desired_tags = [1,2], display_tags=False):
    '''
    Description: Returns the tags that are detected at all times in a video
    Inputs: 
        - video_file: path to video 
        - detector: AprilTag detector
        - display_tags: display detected tags on each frame of the video
    Outputs: 
        - detected_tags: tag IDs that are detected for each frame in the video
        - num_tags: number of tags in each frame of the video 
        - time: timesstamp in video when tags are detected
    '''

    # Read video and set data structures
    cap = cv2.VideoCapture(video_file)
    # num_frames = cv2.CAP_PROP_FRAME_COUNT
    num_detections = []
    vid_time = []
    detected_tags = []


    # Get frames from video and analyze
    max_iters = 100
    idx = 0
    leader_tag = desired_tags[0]
    follower_tag = desired_tags[1]
    leader_detections = 0
    follower_detections = 0
    # while True and idx < max_iters:
    while True:
        is_read, frame = cap.read()
        # cv2.imwrite('frame.png', frame)
        # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # plt.show()
        # Break loop if no more frames remain
        if not is_read:
            break
        
        frame_color = np.copy(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect tags, store number of detections, tag IDs and associated time
        # tags = detector.detect(frame, camera_params=camera_matrix, tag_size=0.0508)
        tags = detector.detect(frame)

        # Only analyze tags with high decision margin
        tags_accurate = []
        # draw_tags(frame, tags)
        for tag in tags:
            if tag.decision_margin > 20:
                tags_accurate.append(tag)
            if tag.tag_id == leader_tag:
                leader_detections = leader_detections + 1
            if tag.tag_id == follower_tag:
                follower_detections = follower_detections + 1
        tags = tags_accurate.copy()

        num_detections.append(len(tags))
        detected_tags.append(tags)
        vid_time.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        # Display image with tags
        if display_tags:
            draw_tags(frame_color, tags)

        idx = idx + 1
    cap.release()

    print(f'Leader Detection Rate: {100*leader_detections/idx}')
    print(f'Follower Detection Rate: {100*follower_detections/idx}')

    num_detections = np.array(num_detections)
    vid_time = np.array(vid_time)
    return detected_tags, num_detections, vid_time/1000

def get_ros_tags(tag_df, detector, desired_tags, display_tags=False):
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    '''
    Description: gets tags for images in ROS format
    Inputs:
        - tag_df: pandas dataframe of tags (containing time and image matrix)
        - detector: AprilTag detector
        - desired_tags: array of desired tag IDs
    '''
    images = tag_df['imgs']
    times = tag_df['time'].to_numpy()
    num_detections = []
    vid_time = []
    detected_tags = []
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 100

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Circularity filtering
    params.filterByCircularity = True
    params.minCircularity = 0.6

    blob_detector = cv2.SimpleBlobDetector_create(params)
    qr_detector = cv2.QRCodeDetector()

    for idx, frame in enumerate(images):
        # Detect tags, store number of detections, tag IDs and associated time
        frame = frame[40:440, 140:550, :]
        frame_color = np.copy(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.bitwise_not(frame)


        tags = detector.detect(frame)
        keypoints = blob_detector.detect(frame)
        blank = np.zeros((1,1))
        blobs = cv2.drawKeypoints(frame, keypoints, blank, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        cv2.namedWindow('all blobs', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('all blobs', 900, 600)
        cv2.imshow('all blobs', blobs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        num_detections.append(len(tags))
        detected_tags.append(tags)
        vid_time.append(times[idx])

    num_detections = np.array(num_detections)
    vid_time = np.array(vid_time)
    return detected_tags, num_detections, vid_time/1000

def get_ros_blobs(tag_df, display_tags=False):
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    '''
    Description: gets tags for images in ROS format using OpenCV blobs
    Inputs:
        - tag_df: pandas dataframe of tags (containing time and image matrix)
    '''
    images = tag_df['imgs']
    times = tag_df['time'].to_numpy()
    num_detections = []
    vid_time = []
    detected_tags = []
    params = cv2.SimpleBlobDetector_Params()

    # params.filterByColor = True
    # params.blobColor = 0

    params.filterByArea = True
    params.minArea = 70

    # Change thresholds
    params.minThreshold = 5
    params.maxThreshold = 215

    # Circularity filtering
    # params.filterByCircularity = True
    # params.minCircularity = 0.1

    blob_detector = cv2.SimpleBlobDetector_create(params)

    for idx, frame in enumerate(images):
        # Detect tags, store number of detections, tag IDs and associated time
        fresize = [40, 440, 140, 550]

        frame = frame[fresize[0]:fresize[1], fresize[2]:fresize[3], :]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.bitwise_not(frame)

        keypoints = blob_detector.detect(frame)

        # pts = np.float([cv2.KeyPoint_convert(keypoints)])
        # pts = pts + 
        # convert coordinates back 
        # keypoints_copy = copy.copy(keypoints)
        for keypoint in keypoints:
            keypoint.pt = (keypoint.pt[0] + fresize[2], keypoint.pt[1] + fresize[0])

        num_detections.append(len(keypoints))
        detected_tags.append(keypoints)
        vid_time.append(times[idx])
        
        # Display image with tags
        if display_tags:
            blank = np.zeros((1,1))
            blobs = cv2.drawKeypoints(frame, keypoints, blank, (0,0,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            cv2.namedWindow('all blobs', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('all blobs', 900, 600)
            cv2.imshow('all blobs', blobs)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    num_detections = np.array(num_detections)
    vid_time = np.array(vid_time)
    return detected_tags, num_detections, vid_time/1000

if __name__=='__main__':
    '''
    Description: Script to detect AprilTags in image using simple detection methods
    '''

    # IMAGE DETECTION
    # Get path to image
    # folder = "../thunderhill/run5_tandem/dt2e-5/"
    # image_name = "image_1.png"
    # filepath_png = os.path.join(folder,image_name)
    # img = cv2.imread(filepath_png)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize detector and draw detected tags
    # detector = dt_apriltags.Detector(families="tag36h11")
    # tags = detector.detect(img_gray)

    # print(f'Number of detected tags: {len(tags)}')
    # draw_tags(img, tags)

    # VIDEO DETECTION
    # Get path to video 
    # folder = "../data/thunderhill/11-16-2022/drone_footage/"
    # vid_name = "DJI_0018.MOV"
    folder = "../data/thunderhill/11-15-2022/run1/"
    vid_name = "DJI_0014.MOV"
    filepath_vid = os.path.join(folder, vid_name)

    detector = dt_apriltags.Detector(families="tag16h5", quad_sigma=0.8)
    detected_tags, num_detections, vid_time = get_video_tags(filepath_vid, detector, display_tags=False)

    # save data to pickle files
    tags_file = os.path.join(folder,'detected_tags.pickle')
    with open(tags_file,'wb') as f:
        pickle.dump(detected_tags, f)

    time_file = os.path.join(folder,'tags_times.pickle')
    with open(time_file,'wb') as f:
        pickle.dump(vid_time, f)

    plt.figure()
    plt.scatter(vid_time, num_detections)
    plt.xlabel('Time (sec)')
    plt.ylabel('Number of detections')

    plt.figure()
    plt.hist(num_detections,[0,1,2,3])
    plt.xlabel('Number of Detections')
    plt.ylabel('Frequency')
    plt.title('Number of detections in video frames')
    plt.show()