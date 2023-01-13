import cv2
import numpy as np
import pandas as pd
import math
import os

def detect_from_video():
    test_type = '2car_nobuffer'
    vid_name = 'DJI_0218_crop_0130_0330'
    num_frame_ros = 3650 - 50 + 1   

    vidpath = 'drone_footage/' + test_type + '/'
    cap = cv2.VideoCapture(vidpath + vid_name + '.MP4')

    centers = []
    center_header = ['redX [pix]', 'redY [pix]', 'greenX [pix]', 'greenY [pix]', 'ccX [pix]', 'ccY [pix]']



    print('total vid frames: ', int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_increment = int(round(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / num_frame_ros))
    end_frame = num_frame_ros * frame_increment
    print('frame increment: ', frame_increment)


    frame_ind = 0
    next_frame = frame_ind + frame_increment

    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        
        _, frame = cap.read()
        if frame_ind == 0 or frame_ind == next_frame:

            print(frame_ind)
            if frame_ind == next_frame:
                next_frame += frame_increment
            
            frame = cv2.resize(frame, (960, 540))
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            ########################################
            # DEFINE MASKS
            ########################################

            # red color mask
            red_lower = np.array([136, 87,111], np.uint8)
            red_upper = np.array([180, 255, 255], np.uint8)
            red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)

            # green color mask
            green_lower = np.array([25, 100, 100], np.uint8)          
            green_upper = np.array([102, 200, 150], np.uint8)
            green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

            # orange color mask
            orange_lower = np.array([150, 80, 50], np.uint8)
            orange_upper = np.array([180, 200, 150], np.uint8)
            orange_mask = cv2.inRange(hsv_frame, orange_lower, orange_upper)

            ########################################
            # MASKS
            ########################################

            kernal = np.ones((5, 5), "uint8")

            red_mask = cv2.dilate(red_mask, kernal)
            res_red = cv2.bitwise_and(frame, frame, mask = red_mask)

            green_mask = cv2.dilate(green_mask, kernal)
            res_green = cv2.bitwise_and(frame, frame, mask = green_mask)

            orange_mask = cv2.dilate(orange_mask, kernal)
            res_orange = cv2.bitwise_and(frame, frame, mask = orange_mask)

            
            ########################################
            # Find Contours and bounding boxes
            ########################################
            red_x = np.nan
            red_y = np.nan
            green_x = np.nan
            green_y = np.nan
            cc_x = np.nan
            cc_y = np.nan

            contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area > 100):
                    x, y, w, h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
                    red_x = x + w/2
                    red_y = y + h/2


            contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area > 100):
                    x, y, w, h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
                    green_x = x + w/2
                    green_y = y + h/2
            

            contours, hierarchy = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if(area < 50):
                    x, y, w, h = cv2.boundingRect(contour)
                    if(y > 220 and y < 320 and x > 430 and x < 530):
                        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, "Orange Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255)) 

                        found_center = 1
                        cc_x = x + w/2
                        cc_y = y + h/2

            centers.append([red_x, red_y, green_x, green_y, cc_x, cc_y])

            ########################################
            # Display
            ########################################

            # Program Termination
            cv2.imshow("Multiple Color Detection in Real-TIme", frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break

        frame_ind += 1

    cap.release()
    cv2.destroyAllWindows()

    output_dir = vidpath + vid_name + '_output/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    centers = np.array(centers)
    df = pd.DataFrame(centers, columns=center_header)
    df.to_csv(output_dir + vid_name[0:8]  + '_centers.csv')

def detect_from_frame(frame):
    centers = []
    frame = cv2.resize(frame, (960, 540))
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    ########################################
    # DEFINE MASKS
    ########################################

    # red color mask
    red_lower = np.array([136, 87,111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)

    # green color mask
    green_lower = np.array([25, 100, 100], np.uint8)          
    green_upper = np.array([102, 200, 150], np.uint8)
    green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)

    # orange color mask
    orange_lower = np.array([150, 80, 50], np.uint8)
    orange_upper = np.array([180, 200, 150], np.uint8)
    orange_mask = cv2.inRange(hsv_frame, orange_lower, orange_upper)

    ########################################
    # MASKS
    ########################################

    kernal = np.ones((5, 5), "uint8")

    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(frame, frame, mask = red_mask)

    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(frame, frame, mask = green_mask)

    orange_mask = cv2.dilate(orange_mask, kernal)
    res_orange = cv2.bitwise_and(frame, frame, mask = orange_mask)

    
    ########################################
    # Find Contours and bounding boxes
    ########################################
    red_x = np.nan
    red_y = np.nan
    green_x = np.nan
    green_y = np.nan
    cc_x = np.nan
    cc_y = np.nan

    contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 100):
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            red_x = x + w/2
            red_y = y + h/2


    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 100):
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
            green_x = x + w/2
            green_y = y + h/2
    

    contours, hierarchy = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area < 50):
            x, y, w, h = cv2.boundingRect(contour)
            if(y > 220 and y < 320 and x > 430 and x < 530):
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Orange Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255)) 

                found_center = 1
                cc_x = x + w/2
                cc_y = y + h/2

    centers.append([red_x, red_y, green_x, green_y, cc_x, cc_y])

    ########################################
    # Display
    ########################################

    # Program Termination
    cv2.imshow("Multiple Color Detection in Real-TIme", frame)
    cv2.waitKey(0)
    # if cv2.waitKey(10) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()

