import sys
from datetime import date, datetime, timedelta
from tracemalloc import start
from tqdm import tqdm
import numpy as np
# setting path
sys.path.append('../src')
from os.path import exists
import numpy as np
from apriltag_tools.Detector import Detector 
from cars.Car import Car
from cars.relative_distances import get_ratio, get_relative_distance
from dynamics_models.CV import CV
from plotting.rel_dists import  plot_relative_distances, plot_relative_distances_and_gt
from dynamics_models.CT import CT
import matplotlib.pyplot as plt
import cv2
import rosbag
import cv_bridge
import imageio
from cv2_tools.cv2utils import draw_center, draw_detections, draw_relative, drawbox




def get_rel_distances(bag_filename, MAX_TIME, visualize=True, save_gif=False):
    bag = rosbag.Bag(bag_filename)
    #cap = cv2.VideoCapture(2)


    mtx = np.load("../data/calibration/mtx.npy")
    dist = np.load("../data/calibration/dist.npy")
    roi = np.load("../data/calibration/roi.npy")
    newcameramtx = np.load("../data/calibration/newcameramtx.npy")
    undistort = False
    car1_msg = '/vrpn_client_node/RigidBody02/pose'
    car2_msg = '/vrpn_client_node/RigidBody01/pose'
    rel_dists = []

    feature_det = cv2.SIFT_create()
    car1 = Car(1, "tag16h5", CV(), feature_det=feature_det)
    car2 = Car(0, "tag16h5", CV(), feature_det=feature_det)
    apriltag_and_sift=True
    only_apriltag = True
    car1_gt, car2_gt = None, None
    dt = 1e-5
    start_time = datetime.now()
    camera = '/usb_cam/image_raw'
    bridge = cv_bridge.CvBridge()
    t2 = start_time
    final_images = []
    threshold=60

    ground_truth_rel = []
    ratios_car1 = []
    ratios_car2 = []
    locations_px1 = []
    locations_px2 = []
    for topic, msg, t in bag.read_messages(topics=[camera,
                                 car1_msg, 
                                 car2_msg]):
        
        t1 = datetime.now()
        if t1-t2<timedelta(seconds=dt):
            continue 

        if topic == car1_msg:
            car1_gt = np.array([msg.pose.position.x, msg.pose.position.y])
        elif topic == car2_msg:
            car2_gt = np.array([msg.pose.position.x, msg.pose.position.y])

        if topic == camera:        
            #img = msg.data
            if car1_gt is None or car2_gt is None:
                continue
            img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if undistort:
                dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
                # crop the image
                x, y, w, h = roi
                img = dst[y:y+h, x:x+w]


        else:
            continue
        if cv2.waitKey(1) == ord('q'):
            break
        if t1-start_time >= timedelta(seconds=MAX_TIME):
            break
        #car1.predict(dt)
        #car2.predict(dt)
        

        detect1, detect2 = False, False
        if apriltag_and_sift or (car1.sift_tag is None or car2.sift_tag is None):
            
            if car1.sift_tag is None or car2.sift_tag is None:
                
                detector = Detector(img=img)
                detections = detector.detect(turn_binary=True, units=8, visualize=False, tag_family=car1.tag_family)
                print(detections)
                detect1 = car1.update_state(detections, img, dt)
                detect2 = car2.update_state(detections, img, dt)
            else:
                print("apriltag")
                detect1 = car1.update_state_apriltag(img, dt,  units=1, tag_family=car1.tag_family, threshold=threshold, visualize=visualize)
                detect2 = car2.update_state_apriltag(img, dt,  units=1, tag_family=car1.tag_family, threshold=threshold, visualize=visualize)
            #   

            
        print(detect1, detect2)
        if not (detect1 and detect2):
            

            if not (car1.sift_tag is None or car2.sift_tag is None or only_apriltag): 
                
                if not detect1:
                    detect1 = car1.update_state_sift(img, dt, threshold=threshold)
                if not detect2:
                    detect2 = car2.update_state_sift(img, dt, threshold=threshold)
            

        
        t2 = datetime.now()
        
        #print("Det1 and 2: ", detect1, detect2)
        if detect1 and detect2:
            ratios_car1.append(car1.ratio)
            ratios_car2.append(car2.ratio)
            locations_px1.append(car1.get_center())
            locations_px2.append(car2.get_center())
            #print(car1.get_center(), car2.get_center())
            rel_dist = get_relative_distance(car1, car2)
            rel_dists.append(rel_dist)
            ground_truth_rel.append(car2_gt-car1_gt)

           
        if visualize:
            img = draw_relative(img, [car1, car2])
            final_images.append(img)
            cv2.imshow("frame", img)

    rel_dists = np.array(rel_dists).squeeze()
    print(rel_dists.shape)
    if save_gif:
        imageio.mimsave('../videos/gifs/flightroom1.gif', final_images, fps=60)
    np.save("../data/flightroom_traj/flightroom1", rel_dists)
    return rel_dists, np.array(ground_truth_rel), ratios_car1, ratios_car2, np.array(locations_px1).squeeze(), np.array(locations_px2).squeeze()





    
if __name__ == '__main__':
    rel_dist_filepath="../plots/flightroom1.png"
    bag_filename = '/home/torstein/Stanford/nav_lab/april_tags/data/flight_room_day2/2022-06-09-13-27-15.bag'

    
    rel_dists, ground_truth, ratios_1, ratios_2, loc1, loc2= get_rel_distances(bag_filename, MAX_TIME=15, save_gif=False)
    plot_relative_distances_and_gt(rel_dists, ground_truth, ratios_1, ratios_2,loc1, loc2,  filepath=rel_dist_filepath)
    
    print(rel_dists.shape)
    #plot_locations(np.array([d.center for d in det1s]), np.array([d.center for d in det2s]))
    #plot_imgs(images, det1s, det2s)