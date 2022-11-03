import cv2
import numpy as np
import sys
sys.path.append('../src')
import matplotlib.pyplot as plt
from apriltag_tools.Detector import Detector
print(cv2.__version__)

def get_new_img(img, tag, dim, random_location=False):
    '''
    Description: Places tag in a location on input image
    Inputs:
        - img: Image to place tag onto 
        - tag: Tag to place onto image (input as an array)
        - dim: Dimensions of tag on image (2-by-1 integer tuple)
        - random_location: Whether to place tag onto image randomly
    '''
    H, W, _ = img.shape

    img_with_tag = np.copy(img)
    resized_tag = cv2.resize(tag, dim, interpolation = cv2.INTER_AREA)
    if random_location:
        cy, cx = int(np.random.uniform(low=0,high=H-dim[0])), int(np.random.uniform(low=0,high=W-dim[0]))
    else:
        cy, cx = 500, 500
    
    img_with_tag[cy:cy+dim[0], cx:cx+dim[1]] = resized_tag
    return img_with_tag, (cx, cy)


# resize image
def detect_and_show(image, c, tag_family):
    '''
    Description: Detects AprilTag on the chosen image
    '''
    detector = Detector(img=image)

    # Detects tag on image
    results = detector.detect(tag_family=tag_family)
    image = detector.img

    #print("Results", len(results))

    def rescale(scale_percent):
        '''
        Description: Scales image by a chosen percentage
        '''
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    scale_percent=40
    c = (c[0]*scale_percent/100., c[1]*scale_percent/100.)
    image = rescale(scale_percent=scale_percent)

    print(f"Expected center", c)
    for r in results:
        r.rescale(scale_percent)
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        #print("Tag id", r.tag_id)
        (ptA, ptB, ptC, ptD) = r.corners
        print(ptA, c)
        #print("Hamming", r.hamming)
        #print("Homography", r.homography)
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))
        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (0, 255, 0), 2)
        cv2.line(image, ptB, ptC, (0, 255, 0), 2)
        cv2.line(image, ptC, ptD, (0, 255, 0), 2)
        cv2.line(image, ptD, ptA, (0, 255, 0), 2)
        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 1, (0, 0, 255), -1)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print("[INFO] tag family: {}".format(tagFamily), f"Id: {r.tag_id}")
        # show the output image after AprilTag detection


    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
    
    threshold=10
    if len(results) == 1 :
        # Return if tag was detected, if tag center is close to expected tag center, and if tag is the expected tag
        return True, np.linalg.norm(np.array(c)-np.array(results[0].corners[0]))<=threshold, results[0].tag_id==0 or results[0].tag_id==2
    if len(results)>1:
        return True, False, False

    else:
        # No tags detected 
        return False, False, False

if __name__=='__main__':
    image_filepath = "data/drone_imgs/grass.png"
    #images = ["image49","DJI_0001", "DJI_0002", "DJI_0003"]
    #idx = 3

    img = cv2.imread(image_filepath)
    tag_16h5 = cv2.imread("../data/tag_family_16h5/tag00.png")
    tag_36h11 = cv2.imread("../data/tag_family_36h11/tag36_11_00002.png")

    dim = (100, 100)


    rescalings = 10
    locations_cnt = 10
    detections_16h5 = np.zeros((rescalings))
    detections_36h11 = np.zeros((rescalings))
    misdetections_16h5 = np.zeros((rescalings))
    misdetections_36h11 = np.zeros((rescalings))
    dims = []
    for i in range(rescalings):
        tag_16h5_cnt, tag_36h11_cnt = 0, 0
        misdetections_16h5_cnt, misdetections_36h11_cnt = 0, 0
        dim = (int(dim[0]*0.8), int(dim[1]*0.8))
        dims.append(dim)
        for k in range(locations_cnt):
            print(dim)
            new_img, c = get_new_img(img, tag_16h5, dim=dim, random_location=True)

            det, correct_loc, correct_id = detect_and_show(new_img, c, "tag16h5")
            if det and correct_loc:
                tag_16h5_cnt+=1
            if det and (not correct_id or not correct_loc):
                misdetections_16h5_cnt+=1
            new_img, c = get_new_img(img, tag_36h11, dim=dim, random_location=True)
            det, correct_loc, correct_id = detect_and_show(new_img, c, "tag36h11")
            if det and correct_loc:
                tag_36h11_cnt+=1
            if det and (not correct_id or not correct_loc):
                misdetections_36h11_cnt +=1
        detections_16h5[i] = tag_16h5_cnt
        detections_36h11[i] = tag_36h11_cnt
        misdetections_16h5[i] = misdetections_16h5_cnt
        misdetections_36h11[i] = misdetections_36h11_cnt


    x_axis = np.arange(len(dims))
    # Create bars
    plt.bar(x_axis-0.1, detections_16h5,0.2 ,label="16h5")
    plt.bar(x_axis+0.1, detections_36h11,0.2, label="36h11")
    plt.legend()
    plt.title("Detections")

    # Create names on the x-axis
    plt.xticks(np.arange(len(dims)), [f"{dim[0]}" for dim in dims])

    plt.show()
    plt.bar(x_axis-0.1, misdetections_16h5,0.2 ,label="16h5")
    plt.bar(x_axis+0.1, misdetections_36h11,0.2, label="36h11")
    plt.legend()
    plt.title("Misdetections")
    # Create names on the x-axis
    plt.xticks(np.arange(len(dims)), [f"{dim[0]}" for dim in dims])
    plt.show()


    print(rescalings*locations_cnt)
