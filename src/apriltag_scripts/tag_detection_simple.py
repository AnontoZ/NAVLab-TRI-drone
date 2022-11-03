import dt_apriltags
import cv2
import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
sys.path.append('../src')
from cv2_tools.cv2utils import *

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

def get_video_tags(video_file, detector, camera_matrix = None, display_tags=False):
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
    # while True and idx < max_iters:
    while True:
        is_read, frame = cap.read()
        # Break loop if no more frames remain
        if not is_read:
            break
    
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect tags, store number of detections, tag IDs and associated time
        # tags = detector.detect(frame, camera_params=camera_matrix, tag_size=0.0508)
        tags = detector.detect(frame, camera_params=camera_matrix)

        # Only analyze tags with high decision margin
        tags_accurate = []
        for tag in tags:
            if tag.decision_margin > 70:
                tags_accurate.append(tag)
        tags = tags_accurate.copy()

        num_detections.append(len(tags))
        detected_tags.append(tags)
        vid_time.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        # Display image with tags
        if display_tags:
            draw_tags(frame, tags)

        idx = idx + 1
    cap.release()

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
    # folder = "../thunderhill/run5_tandem/"
    # vid_name = "DJI_0009.MOV"
    folder = '../data/flight_room/2022-10-26-15-06-35/'
    vid_name = "DJI_0009.MP4"
    filepath_vid = os.path.join(folder, vid_name)

    detector = dt_apriltags.Detector(families="tag16h5")
    cam_matrix = np.array([[2.22655901e+03, 0.00000000e+00, 1.37621969e+03],
        [0.00000000e+00, 2.22722711e+03, 7.41979298e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    detected_tags, num_detections, vid_time = get_video_tags(filepath_vid, detector, camera_matrix=cam_matrix, display_tags=True)

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