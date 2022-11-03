
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../src')
from apriltag_tools.Detector import Detector
print(cv2.__version__)

def get_new_img(img, tag, dim: tuple(int), random_location: bool=False, blur: bool=False, whiteout: bool=False):
    '''
    Description: Places tag in a random location on input image
    Inputs:
        - img: Image to place tag onto 
        - tag: Tag to place onto image (input as an array)
        - dim: Dimensions of tag on image (2-by-1 integer tuple)
        - random_location: Whether to place tag onto image randomly
        - blur: Whether to blur tag
        - whiteout: Whiteout
    '''
    H, W, _ = img.shape

    img_with_tag = np.copy(img)
    resized_tag = cv2.resize(tag, dim, interpolation = cv2.INTER_AREA)

    # 1 in 4 is gaussian blur
    if blur:
        resized_tag = cv2.blur(resized_tag, (3, 3))
    assert(dim[0]==dim[1])

    # Determine location to place tag onto image
    if random_location:
        cy, cx = int(np.random.uniform(low=2*dim[0],high=H-2*dim[0])), int(np.random.uniform(low=2*dim[1],high=W-2*dim[1]))

    else:
        cy, cx = 500, 500
    if dim[0]%2==0:
        high = dim[1]//2
    else: 
        high = dim[1]//2+1

    img_with_tag[cy-dim[0]//2:cy+high, cx-dim[1]//2:cx+high] = resized_tag

    if whiteout:
        img_with_tag[cy:cy+1, cx-dim[1]//2:cx+high//2] = np.random.randint(low=0, high=255, size=(img_with_tag[cy:cy+1, cx-dim[1]//2:cx+high//2]).shape)

    return img_with_tag, (cx, cy)

# resize image
def detect_and_show(image, c, tag_family, tag_id):
    '''
    Description: Detects AprilTag on the chosen image
    '''
    detector = Detector(img=image, tags = [tag_id])

    results = detector.detect(turn_binary = False, tag_family=tag_family)
    
    threshold = 5
    if len(results)==1:
        
        close = np.linalg.norm(np.array(c)-np.array(results[0].center))<threshold
        correct_id = results[0].tag_id == tag_id
    
    image = detector.img
    #image = draw_detections(image, results)
    return len(results)==1 and close and correct_id
    
if __name__=='__main__':
    image_filepath = "../data/drone_imgs/grass.png"

    img = cv2.imread(image_filepath)
    id_cnt = 5

    tags = [cv2.imread("../data/tag_family_16h5/tag0"+str(i)+".png") for i in range(id_cnt)]

    dims = np.array([ (15, 15), (14, 14),(13, 13), (12, 12), (11, 11)])
    dims = np.array([ (20, 20), (19, 19),(18, 18), (17, 17), (16, 16)])
    locations_cnt = 50
    detections = np.zeros((id_cnt, len(dims)))

    for i, dim in tqdm(enumerate(dims)):
        for id, tag in enumerate(tags):
            tag_16h5_cnt=0
            for k in range(locations_cnt):
                blur = False
                whiteout = False
                if k%10 == 0:
                    whiteout=True
                if k%10 == 1:
                    blur=True
                if k%10 == 2:
                    whiteout, blur = True, True
                new_img, c = get_new_img(img, tag, dim=dim, random_location=True, blur=blur, whiteout=whiteout)

                det = detect_and_show(new_img, c, tag_family = "tag16h5", tag_id=id)
                if det:
                    tag_16h5_cnt+=1
            #if det and (not correct_id or not correct_loc):
            #    misdetections_16h5_cnt+=1
            
            detections[id, i] = tag_16h5_cnt

    x_axis = np.arange(len(dims))
    # Create bars
    w = 0.05
    for id, det in enumerate(detections):
        plt.bar(x_axis+id*2*w, det, w ,label=f"tag: {id}")
    plt.legend()
    plt.title("Detections")

    # Create names on the x-axis
    plt.xticks(np.arange(len(dims)), [f"{dim[0]}" for dim in dims])


    plt.show()


