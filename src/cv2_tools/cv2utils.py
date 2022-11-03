import cv2
import numpy as np
def drawbox(img, corners):
    ptA, ptB, ptC, ptD = corners
    print(ptA)
    #for i, corner in enumerate(corners):
    #    cv2.putText(img, str(i),corner, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    cv2.line(img, ptA, ptB, (0, 255, 0), 2)
    cv2.line(img, ptB, ptC, (0, 255, 0), 2)
    cv2.line(img, ptC, ptD, (0, 255, 0), 2)
    cv2.line(img, ptD, ptA, (0, 255, 0), 2)
def rescale(img, scale_percent):

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    return image

def fit_to_screen(img):
    h, w = img.shape[:2]

    max_height = 900.
    max_width = 2000.

    scale_percent = min(100*max_height/h, 100*max_width/w)
    img = rescale(img, scale_percent)
    return img, scale_percent


def draw_center(image, cars):
    for car in cars:
        c = car.get_center().astype(int).flatten()
        print(c)
        cv2.circle(image,c, 3, (0, 0, 255), -1)
    return image

def draw_relative(image, cars, thickness=3):
    c1 = cars[0].get_center().astype(int).flatten()
    c2 = cars[1].get_center().astype(int).flatten()

    cv2.line(image, c1, c2, color=(0, 255, 0), thickness=thickness)
    return image
def draw_detections(image, results, extra_text="", rescale=True, low_x=0, low_y=0):
    
    for r in results:
        print(len(r.corners))
        for i, corner in enumerate(r.corners):
            r.corners[i] += np.array([low_x, low_y])
        drawbox(image, r.corners)
        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
        tag_id = r.tag_id
        cv2.putText(image, str(r.tag_id), r.corners[0], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)
        #cv2.putText(image, tagFamily+" Id:"+str(tag_id), (r.corners[0][0], r.corners[0][1] - 15),
        #    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
        #print("[INFO] tag family: {}".format(tagFamily), "tag_id", tag_id)

    cv2.putText(image, extra_text, (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    if rescale:
        image, _ = fit_to_screen(image)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

def undistort(src):
    width  = src.shape[1]
    height = src.shape[0]

    distCoeff = np.zeros((4,1),np.float64)


    # TODO: add your coefficients here!
    k1 = -1.0e-7; # negative to remove barrel distortion
    k2 = -1.0e-7#-1.0e-9;
    p1 = 1e-5;
    p2 = 1e-5

    distCoeff[0,0] = k1;
    distCoeff[1,0] = k2;
    distCoeff[2,0] = p1;
    distCoeff[3,0] = p2;

    # assume unit matrix for camera
    cam = np.eye(3,dtype=np.float32)

    cam[0,2] = width/2.0  # define center x
    cam[1,2] = height/2.0 # define center y
    cam[0,0] = 10.        # define focal length x
    cam[1,1] = 10.        # define focal length y

    # here the undistortion will be computed  
    dst = cv2.undistort(src,cam,distCoeff)
    return dst