import os
import rosbag
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv_bridge

if __name__ == '__main__':
    '''
    Description: Script to get data from flight room test on 10/26/2022
    '''
    # Get filenames, partition into test set and verification set 
    bag_filenames = ['2022-10-26-15-06-35.bag','2022-10-26-15-17-15.bag']
    files_test = bag_filenames[0:2]
    files_verify = bag_filenames[2:]
    bag_folder_name = '../data/flight_room/'
    selected_file = bag_filenames[0]
    bag_filename = os.path.join(bag_folder_name,selected_file)

    # Open rosbag and save data as csv
    bag = rosbag.Bag(bag_filename,'r')
    list_topics = bag.get_type_and_topic_info()[1].keys()
    list_topics = list(list_topics)
    list_fnames = ['cone_01.csv','cone_02.csv','cone_03.csv','drone.csv','turtlebot_01.csv','turtlebot_02.csv']
    idx_name = 0
    # types = []
    # for i in range(0,len(bag.get_type_and_topic_info()[1].keys())):
    #     types.append(bag.get_type_and_topic_info()[1].values()[i][0])
    for idx in range(0,len(list_topics)):
        topic_curr = list_topics[idx]
        csv_name = list_fnames[idx]
        list_poses = []
        for topic, msg, t in bag.read_messages(topics=topic_curr):
            pos = msg.pose.position
            quat = msg.pose.orientation
            list_poses.append([t.to_time(), pos.x,pos.y,pos.z,quat.x,quat.y,quat.z,quat.w])
            poses = np.array(list_poses)
        np.savetxt(os.path.join(bag_folder_name,selected_file[:-4],csv_name),poses)
