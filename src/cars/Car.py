import numpy as np
from cv2_tools.cv2utils import draw_detections, drawbox
from sift.retrieve_center import AprilTagSift
from cars.relative_distances import get_ratio
from sift.retrieve_center import get_center_w_sift, AprilTagSift
from apriltag_tools.Detector import Detector
import cv2
class Car:
    def __init__(self, tag_id, tag_family, dynamic_model, size=9e-2, feature_det = None):
        '''
        Description: Creates a new Car object
        Inputs:
            - tag_id: tags associated with car 
            - tag_family: tag family associated with car
            - dynamic_model: vehicle dynamics
            - size: size of car TODO: confirm
            - feature_det: Type of feature detector to use
        '''
        ## Pixel pos and velocity
        self.state = np.zeros((4, 1))
        self.trajectory = []
        self.dyn_model = dynamic_model
        self.tag_id = tag_id
        self.tag_family = tag_family
        self.sift_tag = None
        self.size=size
        self.feature_det = feature_det

    def predict(self, dt):
        '''
        Description: predicts vehicle state after a given time step 
        '''
        self.state = self.dyn_model.f(self.state, None, dt)

    def update_state(self, detections, img, dt, low_x = 0, low_y = 0):
        '''
        Description: updates vehicle state after tag is detected 
        '''
        det = [det for det in detections if det.tag_id == self.tag_id]
        for detect in detections:
            print(detect.tag_id, self.tag_id)
        if not len(det):
            return False
        self.detection = det[0]
        if self.sift_tag is None :
            self.sift_tag = AprilTagSift(self.detection, self.feature_det, img)


        new_c = self.detection.center.reshape((-1, 1)) + np.array([low_x, low_y]).reshape((-1, 1))
        self.set_ratio()

        self.update_trajectory(new_c, dt)
        return True

    def set_ratio(self):
        #self.ratio=0.005#495325200310615
        #return 
        px_dist = np.median(np.linalg.norm(np.array([
            self.detection.corners[1]-self.detection.corners[0], 
            self.detection.corners[2]-self.detection.corners[1], 
            self.detection.corners[3]-self.detection.corners[2],
            self.detection.corners[0]-self.detection.corners[3]
        ]).squeeze(), axis=1))
        
        self.ratio = self.size/px_dist
        
        #radial correction
        #radial_correction = np.abs(self.get_center()[0])/17000+ np.abs(self.get_center()[1])/4000
        #self.ratio*= (1-radial_correction)
        
    def update_state_apriltag(self, img, dt, units=1, tag_family="tag16h5", threshold=400, visualize=False):
        search_area, low_x, low_y = self.get_search_area(img, threshold)

        detector = Detector(img = search_area)
        detections = detector.detect(turn_binary=True, units=units, tag_family=tag_family)
        if visualize:
            draw_detections(img, detections, rescale=False, low_x=low_x, low_y=low_y)
        return self.update_state(detections, img, dt, low_x=low_x, low_y=low_y)

    def get_search_area(self, img, threshold):
        c = self.get_center().flatten()
        
        low_x, low_y = c.flatten() - threshold*np.ones((2, ))
        high_x, high_y = c.flatten() + threshold*np.ones((2, ))
        
        low_x = min(int(max(0, low_x)), img.shape[1])
        high_x = min(int(max(0, high_x)), img.shape[1])
        low_y = min(int(max(0, low_y)), img.shape[0])
        high_y = min(int(max(0, high_y)), img.shape[0])
        
        if high_y-low_y<=50:
            low_y -= threshold
            high_y += threshold
            
        if high_x-low_x<=50:
            low_x -= threshold
            high_x += threshold
        low_x = min(int(max(0, low_x)), img.shape[1])
        high_x = min(int(max(0, high_x)), img.shape[1])
        low_y = min(int(max(0, low_y)), img.shape[0])
        high_y = min(int(max(0, high_y)), img.shape[0])
        search_area = img[low_y:high_y, low_x:high_x]

        try:
            assert(search_area.shape[0]>50 and search_area.shape[1] > 50), search_area.shape
        except Exception as e:
            print(e)

        return search_area, low_x, low_y
            
    def update_state_sift(self, img, dt, threshold=400):
        search_area, low_x, low_y = self.get_search_area(img, threshold)

        new_c, detect = get_center_w_sift(search_area, self.sift_tag)
    
        if not detect or np.any(np.isnan(new_c)):
            #cv2.imshow("frame", img)
            #cv2.waitKey(0)
            return False
        else:
            assert(new_c.shape==(2,))
            new_c += np.array([low_x, low_y])

            self.update_trajectory(new_c, dt)

            return True

    def update_trajectory(self, c, dt):
        upd_state = np.vstack((
            c.reshape((-1, 1)),
            self.state[2:].reshape((-1, 1))
        ))
        if len(self.trajectory)>1:
            upd_state[2:] = (c.reshape((-1, 1))-self.state[:2])/dt
        self.trajectory.append(upd_state)
        self.state = self.trajectory[-1]


    def get_center(self):
        return self.state[:2]

    