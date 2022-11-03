import rosbag
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv_bridge
import glob

if __name__ == '__main__':
    bag_filename = '/home/torstein/Stanford/nav_lab/flightroom/2022-05-27-11-38-06.bag'
    bag = rosbag.Bag(bag_filename)
    camera = '/usb_cam/image_raw'
    bridge = cv_bridge.CvBridge()
    body1 =  '/vrpn_client_node/RigidBody01/pose'
    poses = []
    img_array = []
    N = 1000
    n = 0
    for topic, msg, t in bag.read_messages(topics=[camera,
                                    '/vrpn_client_node/RigidBody01/pose', 
                                 '/vrpn_client_node/RigidBody02/pose']):
        if topic == camera:        
            #img = msg.data
            n +=1 

            if n%10:
                continue
            img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            #print(img.shape)
            cv2.imshow('frame', img)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
            if cv2.waitKey(1) == ord('q'):
                break
            
            if n == N:
                break




    out = cv2.VideoWriter('data/flightroom_traj/videos/video1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
            
