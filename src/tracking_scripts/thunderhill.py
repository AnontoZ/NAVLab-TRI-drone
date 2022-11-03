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
from cv2_tools.cv2utils import draw_center, draw_relative, fit_to_screen, rescale

from dynamics_models.CV import CV
from dynamics_models.CT import CT
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import imageio

def get_rel_distances(folder_in, visualize=True):

    rel_dists = []

    N = 200

    feature_det = cv2.SIFT_create()
    tag_family="tag36h11"
    car1 = Car(0, tag_family, CV(), feature_det=feature_det)
    car2 = Car(2, tag_family, CV(), feature_det=feature_det)
    apriltag_and_sift=False
    only_apriltag =False

    dt = 0.1
    images = []
    start_idx = 1
    for i in tqdm(range(start_idx, start_idx+N+1)):
        car1.predict(dt)
        car2.predict(dt)
        t1 = datetime.now()
        filepath = folder_in+"image_"+str(i)+".png"
        
        if not exists(filepath):
            print(filepath, "does not exist. Quitting")
            break
        img = cv2.imread(filepath)#, cv2.IMREAD_GRAYSCALE)
        print("image reading", datetime.now()-t1)
        detect1, detect2 = False, False
        search_threshold = 100
        if apriltag_and_sift or (car1.sift_tag is None or car2.sift_tag is None):
            
            if car1.sift_tag is None or car2.sift_tag is None:
                detector = Detector(img=img)
                detections = detector.detect(turn_binary=True, tag_family=tag_family)
                detect1 = car1.update_state(detections, img, dt)
                detect2 = car2.update_state(detections, img, dt)
            else:
                detect1 = car1.update_state_apriltag(img, dt, tag_family=tag_family, threshold=search_threshold)
                detect2 = car2.update_state_apriltag(img, dt, tag_family=tag_family, threshold=search_threshold)
            #


        if not (detect1 and detect2):

            if car1.sift_tag is None or car2.sift_tag is None or only_apriltag: 
                continue
            if not detect1:
                detect1 = car1.update_state_sift(img, dt, threshold=search_threshold)
            if not detect2:
                detect2 = car2.update_state_sift(img, dt, threshold=search_threshold)
           
        if detect1 and detect2:
            img = draw_center(img, [car1, car2])
            
            rel_dist = get_relative_distance(car1, car2)
            rel_dists.append(rel_dist)
        if visualize:
            draw_relative(img, [car1, car2], thickness=15)
            img_to_show, _ = fit_to_screen(img)
            img_to_show = rescale(img_to_show, 40)
            images.append(img_to_show)
            #cv2.imshow("frame", img_to_show)
            #cv2.waitKey(1)
        t2 = datetime.now()
        print(t2-t1)
    # print("One: ", one_detections, "avg: ", one_detections/idx)
    rel_dists = np.array(rel_dists)
    print(len(rel_dists))
    if visualize:
        imageio.mimsave('../videos/gifs/thunderhill_1.gif', images, fps=60)

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
    folder_in = "../thunderhill/run5_tandem/dt2e-1/"
    assert(folder_in[-1] == "/")
    rel_dist_filepath = "../thunderhill/plots/run5.png"
    
    rel_dists = get_rel_distances(folder_in=folder_in)
  
    plot_relative_distances(rel_dists, filepath=rel_dist_filepath)
    #plot_locations(np.array([d.center for d in det1s]), np.array([d.center for d in det2s]))
    #plot_imgs(images, det1s, det2s)