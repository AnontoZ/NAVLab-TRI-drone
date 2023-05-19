from base64 import decode
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
from apriltag_tools.ImageParser import parse_img

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

def get_video_tags_windows(video_file, detector, desired_tags = [1,2], display_tags=False, turn_binary=True):
    '''
    Description: Returns the tags that are detected at all times in a video using a windowing method
    Inputs: 
        - video_file: path to video 
        - detector: AprilTag detector
        - display_tags: display detected tags on each frame of the video
        - turn_binary: Convert image to binary
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
    max_iters = 500
    idx = 0
    leader_tag = desired_tags[0]
    follower_tag = desired_tags[1]
    leader_detections = 0
    follower_detections = 0
    all_results = []
    while True and idx < max_iters:
    # while True:
        is_read, frame = cap.read()
        # cv2.imwrite('frame.png', frame)
        # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # plt.show()
        # Break loop if no more frames remain
        if not is_read:
            break
      
        frame_color = np.copy(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        window_idx = parse_img(frame, units=4)
        tags_accurate = []
        detected_ids = []

        # iterate through windows 
        for i, idxs in enumerate(window_idx):
            # Can't have more than 4 tags in image
            if len(tags_accurate) == 4:
                break

            l, r, t, b = idxs
            window = frame[t:b, l:r]

            # Checks whether to turn image to binary
            if turn_binary:
                ret, window = cv2.threshold(window, 0.75*(np.max(window)-np.min(window)), 255, cv2.THRESH_BINARY)
            
            # Detects AprilTags in window 
            tags = detector.detect(frame)
            
            # Sees whether detected tags correspond to expectations 
            for tag in tags:
                if tag.decision_margin > 5 and tag.tag_id in desired_tags and tag.tag_id not in detected_ids:
                    # Adjust center and corners
                    for i in range(len(tag.corners)):
                        tag.corners[i] = [tag.corners[i][0]+l, tag.corners[i][1] + t]
                    tag.center = np.array([tag.center[0]+l, tag.center[1] + t], dtype=np.int16)    # Center of AprilTag in image frame


                    tags_accurate.append(tag)
                    detected_ids.append(tag.tag_id)
                    if tag.tag_id == leader_tag:
                        leader_detections = leader_detections + 1
                    if tag.tag_id == follower_tag:
                        follower_detections = follower_detections + 1
            tags = tags_accurate.copy()

            num_detections.append(len(tags))
            detected_tags.append(tags)
            vid_time.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            
            # Visualize results, show image to user 
            if display_tags:
                cv2.imshow('frame meas', draw_tags(frame_color, tags))
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        idx = idx + 1
    cap.release()

    print(f'Leader Detection Rate: {100*leader_detections/idx}')
    print(f'Follower Detection Rate: {100*follower_detections/idx}')

    num_detections = np.array(num_detections)
    vid_time = np.array(vid_time)
    return detected_tags, num_detections, vid_time/1000

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
    # max_iters = 500
    max_iters = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 20
    idx = 0
    leader_tag = desired_tags[0]
    follower_tag = desired_tags[1]
    leader_detections = 0
    follower_detections = 0
    leader_arr = np.zeros(max_iters)
    follower_arr = np.zeros(max_iters)
    while True and idx < max_iters:
    # while True:
        is_read, frame = cap.read()
        # cv2.imwrite('frame.png', frame)
        # plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # plt.show()
        # Break loop if no more frames remain
        if not is_read:
            print("Video not read")
            break
      
        frame_color = np.copy(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # windows = parse_img(frame, units=8)

        # Detect tags, store number of detections, tag IDs and associated time
        # tags = detector.detect(frame, camera_params=camera_matrix, tag_size=0.0508)
        tags = detector.detect(frame)

        # Only analyze tags with high decision margin
        tags_accurate = []
        # draw_tags(frame, tags)
        for tag in tags:
            if tag.decision_margin > 10:
            # if tag.decision_margin > 10:
                # print(tag.decision_margin)
                # print("\n")
                tags_accurate.append(tag)
                if tag.tag_id == leader_tag:
                    leader_detections = leader_detections + 1
                    leader_arr[idx] = 1
                if tag.tag_id == follower_tag:
                    follower_detections = follower_detections + 1
                    follower_arr[idx] = 1
        # if display_tags and len(tags_accurate) < 2:
        #     cv2.namedWindow("Undetected AprilTags", cv2.WINDOW_NORMAL)
        #     cv2.resizeWindow("Undetected AprilTags", 900, 600)
        #     cv2.imshow("Undetected AprilTags", frame_color)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        tags = tags_accurate.copy()

        num_detections.append(len(tags))
        detected_tags.append(tags)
        vid_time.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        # Display image with tags
        # if len(tags_accurate) > 1:
        if display_tags:
            pass
            draw_tags(frame_color, tags)

        idx = idx + 1
    cap.release()

    print(f'Leader Detection Rate: {100*leader_detections/idx}')
    print(f'Follower Detection Rate: {100*follower_detections/idx}')

    leader_arr = np.cumsum(leader_arr)
    follower_arr = np.cumsum(follower_arr)
    frame_count = np.arange(1, max_iters+1)

    vid_time = np.array(vid_time)

    fig, ax = plt.subplots()
    ax.plot(vid_time/1000, 100*leader_arr/frame_count, label='Leader')
    ax.plot(vid_time/1000, 100*follower_arr/frame_count, label='Follower')
    ax.set_ylabel('Detection Rate (%)')
    ax.set_xlabel('Time (sec)')
    ax.legend()
    ax.grid()
    plt.show()

    num_detections = np.array(num_detections)
    # vid_time = np.array(vid_time)
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

def plot_tags(tags_list, leader_id = 1, follower_id = 2):
    '''
    Description: Plots an array of tags 
    '''
    leader_pts = []
    follower_pts = []
    for tags in tags_list:
        for tag in tags:
            if tag.tag_id == leader_id:
                leader_pts.append(tag.center)
            if tag.tag_id == follower_id: 
                follower_pts.append(tag.center)

    leader_pts = np.asarray(leader_pts)
    follower_pts = np.asarray(follower_pts)

    fig, ax = plt.subplots()
    ax.scatter(leader_pts[:,0], leader_pts[:,1], label='Leader', color='red')
    ax.scatter(follower_pts[:,0], follower_pts[:,1], label='Follower ', color='blue')
    ax.legend()
    ax.axis('equal')
    ax.set_xbound(0, 3840)
    ax.set_ybound(0, 2160)    
    plt.gca().invert_yaxis()
    ax.grid()

    fig, ax = plt.subplots()
    ax.scatter(follower_pts[:,0], follower_pts[:,1], label='Follower ', color='blue')
    ax.legend()
    ax.set_xbound(0, 3840)
    ax.set_ybound(0, 2160)
    ax.axis('equal')
    ax.grid()
    plt.gca().invert_yaxis()
    plt.show()

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
    # folder = "../data/thunderhill/11-15-2022/run1/"
    # vid_name = "DJI_0016.MOV"
    # vid_name = "DJI_0014_30.mp4"
    folder = "../data/thunderhill/"
    vid_name = "20_mph_run3.MOV"
    # vid_name = "10mph_run3.MOV"
    filepath_vid = os.path.join(folder, vid_name)

    # detector = dt_apriltags.Detector(families="tag16h5", quad_sigma=4, decode_sharpening=1)
    # detector = dt_apriltags.Detector(families="tag16h5", quad_sigma=0.8, decode_sharpening=0.5)
    detector = dt_apriltags.Detector(families="tag36h11", decode_sharpening=0.5)
    # detector = dt_apriltags.Detector(families="tag36h11", quad_sigma=0.8)
    # detected_tags, num_detections, vid_time = get_video_tags_windows(filepath_vid, detector, display_tags=False)
    detected_tags, num_detections, vid_time = get_video_tags(filepath_vid, detector, desired_tags=[0, 2], display_tags=False)
    plot_tags(detected_tags, 0, 2)

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