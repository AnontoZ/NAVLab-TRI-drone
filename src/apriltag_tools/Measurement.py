import cv2
import numpy as np
import apriltag

class Measurement():
    def __init__(self, raw_img,  tag_family="tag36h11"):
        '''
        Description: Creates a Measurement object 
        Inputs:
            - raw_img: Image for analysis
            - tag_family: AprilTag family in image
        '''
        self.raw_img=raw_img
        self.img = np.copy(raw_img)
        self.H, self.W = self.raw_img.shape[:2]
        options = apriltag.DetectorOptions(families=tag_family,
                                #quad_decimate=1.0,
                                #quad_blur=0.8
                                #refine_pose=5.0
                                )
        self.detector = apriltag.Detector(options)

    def grayscale(self):
        '''
            Description: Converts image to grayscale using openCV
        '''
        if len(self.img.shape)>2:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def scale(self, scale_percent):
        '''
        Description: Resize img based on a given scale
        Inputs:
            - scale_percent: percent to scale img by 
        '''
        width = int(self.W * scale_percent / 100)
        height = int(self.H * scale_percent / 100)
        dim = (width, height)
        self.img = cv2.resize(self.raw_img, dim, interpolation = cv2.INTER_AREA)

    def turn_binary_const(self, threshold):
        '''
        Description: Turn img into a binary image (black-and-white) based on a chosen pixel threshold
        Input:
            - threshold: threshold to determine pixel color (pixels > threshold are white)
        '''
        self.img = np.array(self.img>threshold, dtype=np.uint8)*255

    def turn_binary(self):
        '''
        Description: Turn img into a binary image (black-and-white)
        '''
        threshold = 0.75*(np.max(self.img)-np.min(self.img))
        self.img = np.array(self.img>threshold, dtype=np.uint8)*255

    def detect(self):
        '''
        Description: Finds AprilTags in img 
        '''
        self.results = self.detector.detect(self.img)
        return self.results

    def reset(self):
        '''
        Description: Resets img to original input image
        '''
        self.img = np.copy(self.raw_img)

    def imadjust(self):
        raise NotImplemented
        #self.img = imadjust(self.img, tol=2)

    def turn_binary_adaptive(self):
        '''
        Description: Turns img into a binary image (black-and-white) along a 4x4 grid
        '''
        vertical_counts =  4
        horizontal_counts = 4
        self.binary_img = np.zeros_like(self.img, dtype=np.uint8)
        dx = self.img.shape[1]//vertical_counts
        dy = self.img.shape[0]//horizontal_counts
        for i in range(vertical_counts):
            for j in range(horizontal_counts):
                # Find boundaries of box 
                t, b, l, r = i*dy,(i+1)*dy, j*dx,(j+1)*dx

                # Find threshold for pixels and convert to binary
                threshold = min(220, int(np.quantile(self.img[t:b, l:r], 0.99)))
                threshold = 0.75*(np.max(self.img[t:b, l:r])-np.min(self.img[t:b, l:r]))
                self.binary_img[t:b, l:r] = np.array(self.img[t:b, l:r]>threshold, dtype=np.uint8)*255
        #cv2.imshow("binary image, adaptive threshold", self.binary_img)
        self.img = self.binary_img

def test_adaptive():
    '''
    Description: Test to see whether adaptive greyscaling works 
    '''
    img = np.array([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1,2, 3]])

    meas = Measurement(img, grayscale=True)
    meas.turn_binary_adaptive(threshold=1)
    print(meas.img)

if __name__=='__main__':
    #test_adaptive()
    #raise Exception

    folder = "../thunderhill/run5_tandem/photos/DJI_0009/"
    image_name = "image_14"
    filepath_png = folder +image_name+".png"

    image = cv2.imread(filepath_png)[600:2000, 500:2000]

    meas = Measurement(image)
    meas.grayscale()
    meas.turn_binary_adaptive()
    res = meas.detect()
    print("Result", len(res))

    cv2.imshow("Image", meas.img)
    cv2.waitKey(0)
 


