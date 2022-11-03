import sys
from datetime import date, datetime
from tqdm import tqdm
# setting path
sys.path.append('../src')
from os.path import exists
import numpy as np
from apriltag_tools.Detector import Detector 
from cars.Car import Car
from cars.relative_distances import get_ratio, get_relative_distance
from dynamics_models.CV import CV
from dynamics_models.CT import CT
import matplotlib.pyplot as plt
import cv2

def get_rel_distances():
    cap = cv2.VideoCapture(2)

    rel_dists = []

    N = 500

    feature_det = cv2.SIFT_create()
    car1 = Car(0, "tag16h5", CV(), feature_det=feature_det)
    car2 = Car(1, "tag16h5", CV(), feature_det=feature_det)
    apriltag_and_sift=True
    only_apriltag = True

    dt = 0.1
    t1  = datetime.now()
    for i in tqdm(range(1, N+1)):
        _, img = cap.read()
        if cv2.waitKey(1) == ord('q'):
            cv2.imshow('frame', img)
            break
        dt = 0.00001
        car1.predict(dt)
        car2.predict(dt)
        t1 = datetime.now()
        

        
        detect1, detect2 = False, False
        if apriltag_and_sift or (car1.sift_tag is None or car2.sift_tag is None):
            
            if car1.sift_tag is None or car2.sift_tag is None:
                
                detector = Detector(img=img)
                detections = detector.detect(turn_binary=True, units=3, visualize=True, tag_family=car1.tag_family)
                print(len(detections))
                print(detections)
                detect1 = car1.update_state(detections, img, dt)
                detect2 = car2.update_state(detections, img, dt)
            else:
                detect1 = car1.update_state_apriltag(img, dt,  units=2, tag_family=car1.tag_family)
                detect2 = car2.update_state_apriltag(img, dt,  units=2, tag_family=car1.tag_family)
            #

        if not (detect1 and detect2):
        

            if not (car1.sift_tag is None or car2.sift_tag is None or only_apriltag): 
                
                if not detect1:
                    detect1 = car1.update_state_sift(img, dt)
                if not detect2:
                    detect2 = car2.update_state_sift(img, dt)
            

        
        t2 = datetime.now()
        print("Det1 and 2: ", detect1, detect2)

        if detect1 and detect2:
            rel_dist = get_relative_distance(car1, car2)
            rel_dists.append(rel_dist)
            print(rel_dist)
        print(t2-t1)
    # print("One: ", one_detections, "avg: ", one_detections/idx)
    rel_dists = np.array(rel_dists)
   

    return rel_dists

def plot_relative_distances(rel_dists, filepath):

    fig = plt.figure()
    plt.plot(rel_dists[:, 0], rel_dists[:, 1])
    plt.savefig(filepath, dpi=fig.dpi)
    plt.show()

def plot_locations(pos1, pos2):
    plt.plot(pos1[:, 0], pos1[:, 1])
    plt.plot(pos2[:, 0], pos2[:, 1])
    plt.show()




    
if __name__ == '__main__':
    rel_dist_filepath = "thunderhill/plots/live.png"
    
    rel_dists = get_rel_distances()
  
    plot_relative_distances(rel_dists, filepath=rel_dist_filepath)
    print(rel_dists.shape)
    #plot_locations(np.array([d.center for d in det1s]), np.array([d.center for d in det2s]))
    #plot_imgs(images, det1s, det2s)