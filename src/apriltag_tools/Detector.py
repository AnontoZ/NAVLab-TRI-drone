#from AprilTag import AprilTag
from apriltag_tools.ImageParser import parse_img
from apriltag_tools.Measurement import Measurement
from typing import Union
import numpy as np
import cv2
#from PyTorchYOLOv3.pytorchyolo.detect import detect
from cv2_tools.cv2utils import drawbox
from datetime import datetime
#from PyTorchYOLOv3.pytorchyolo import detect, models
# model = models.load_model(
#   "PyTorchYOLOv3/config/yolov3.cfg", 
#   "PyTorchYOLOv3/weights/yolov3.weights")
class Result():
    '''
    Description: Result class to store parameters of detected AprilTags
    '''
    def __init__(self, res, l, r, t, b):
        '''
        Description: Creates new Result
        Inputs:
            - res: result from AprilTag detection algorithm
            - l: left pixel in image window
            - r: right pixel in image window
            - t: top pixel in image window
            - b: bottom pixel in image window 
        '''
        self.tag_id = res.tag_id
        self.center = np.array([res.center[0]+l, res.center[1] + t], dtype=np.int16)    # Center of AprilTag in image frame
        self.corners = res.corners
        self.tag_family = res.tag_family

        # Corners of AprilTag in image frame
        for i in range(len(self.corners)):
            self.corners[i] = [self.corners[i][0]+l, self.corners[i][1] + t]
        self.corners = np.array(self.corners, dtype=np.int16)

    def rescale(self, scale_percent):
        '''
        Description: Rescales AprilTag location in image 
        Inputs:
            - scale_percent: percent by which image is scaled 
        '''
        for i in range(len(self.corners)):
            self.corners[i] = np.array(self.corners[i], dtype=np.float16)*0.01*scale_percent
        self.corners = np.array(self.corners, dtype=np.int16)
        self.center = np.array(self.center, dtype=np.float16)*0.01*scale_percent
        self.center = np.array(self.center, dtype=np.int16)

        #self.center[1] *= 0.01*scale_percent
class Detector():
    '''
    Description: class to detect AprilTags in chosen image 
    '''

    def __init__(self, filepath: str=None, img=None, tags=[i for i in range(30)], camera_matrix=np.eye(3)): 
        '''
        Description: Creates a Detector object 
        Input:
            - filepath: path to desired image
            - img: image chosen for analysis
            - tags: AprilTag IDs of tags in image (note this is NOT the AprilTag family)
        '''
        self.tags = tags

        self.camera_matrix = camera_matrix

        if filepath is not None:
            self.img = cv2.imread(filepath)
        else:
            self.img = img

    def detect(self, adaptive_threshold: bool=False, tag_family: str="tag36h11", 
        turn_binary: bool=True, units: int=4, visualize: bool=False):
        '''
        Description: Detects AprilTags in image
        Input:
            - adaptive_threshold: Selects whether img will undergo adaptive grayscaling 
            - tag_family: Indicates tag family found in image
            - turn_binary: Selects whether img will be processed as a binary (black-and-white) image
            - units: Grid size by which image will be processed (units-by-units grid)
            - visualize: Selects whether to show processed image 
        '''
        detections = []
        tags_seen = {}
        # Retuns windows over which image is processed 
        self.image_idxs = parse_img(self.img, units=units)

        start = datetime.now()

        # Iterates through windows 
        for i, idxs in enumerate(self.image_idxs):
            # Can't have more than 10 tags in image
            if len(tags_seen) == 10:
                break

            l, r, t, b = idxs
            window = self.img[t:b, l:r]
            #boxes = detect.detect_image(model, window, conf_thres=0.1)
            #for box in boxes:
                #x1, y1, x2, y2, conf, class_ = box
                #x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                #l, r, t, b = l+x1, l+x2, t+y1, t+y2 


                #new_window = window[y1:y2,x1:x2]
            # Creates new measurement over window and converts to grayscale
            meas = Measurement(window, tag_family=tag_family)
            meas.grayscale()

            # Checks whether to turn image to binary
            if turn_binary:
                if adaptive_threshold:
                    meas.turn_binary_adaptive()
                else:
                    meas.turn_binary()
            
            # Detects AprilTags in image 
            results = meas.detect()
            
            # Sees whether detected tags correspond to expectations 
            for result in results:
                res = Result(result, l, r, t, b)
                tag_id = res.tag_id
                
                if tag_id not in self.tags:
                    #print("unseen apriltag", tag_id)
                    continue
                if tag_id in tags_seen:
                    continue
                tags_seen[tag_id] = True
                detections.append(res)
            
            # Visualize results, show image to user 
            if visualize:
                cv2.imshow('frame meas', meas.img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                #cv2.imshow(f"Image", meas.img)
                #cv2.waitKey(0)
        #print("Timing", datetime.now()-start)
        return detections

    def get_pose(self, detection):
        # pose = apriltag
        raise NotImplemented('Function not implemented')

if __name__ == '__main__':
    folder_in = "thunderhill/run3/mph_10/photos/"
    image_name = "image_1"
    filepath = folder_in + image_name +".jpg"
    folder_out = "thunderhill/run3/mph_10/photos/pngs/"

    detector = Detector(filepath, folder_out)
    detections = detector.detect()
    print(detections)
    
