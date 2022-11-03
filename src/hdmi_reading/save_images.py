import numpy as np
import cv2
import sys
import time
sys.path.append("../april_tags")
cap = cv2.VideoCapture('/dev/video2')
while True:
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    # Capture frame-by-frame

    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    print(gray.shape)
    # Display the resulting frame
    cv2.imshow('frame', gray)
    k = cv2.waitKey(1)
    if k == ord('s'):
        cv2.imwrite("../data/calibration/image_"+str(time.time())+".jpg", gray)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

