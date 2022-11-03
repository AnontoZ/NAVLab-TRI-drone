import rosbag
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv_bridge

if __name__ == '__main__':
    bag_filename = '/home/torstein/Stanford/nav_lab/flightroom/2022-05-27-11-38-06.bag'
    bag = rosbag.Bag(bag_filename)
    camera = '/usb_cam/image_raw'
    bridge = cv_bridge.CvBridge()
    body1 =  '/vrpn_client_node/RigidBody01/pose'
    poses = []
    for topic, msg, t in bag.read_messages(topics=[camera,
                                    '/vrpn_client_node/RigidBody01/pose', 
                                 '/vrpn_client_node/RigidBody02/pose']):

        if topic == camera:        
            #img = msg.data
            img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            #print(img.shape)
            cv2.imshow('frame', img)

            if cv2.waitKey(1) == ord('q'):
                break

        if topic == body1:
            poses.append(np.array([msg.pose.position.x, msg.pose.position.y]))
            print(msg.pose.position)
            #raise Exception
    poses = np.array(poses).squeeze()
    np.save("data/flightroom_traj/flightroom1", poses)
        
    cv2.waitKey(0)