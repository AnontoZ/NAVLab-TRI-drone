from numpy import imag
import cv2
    
def parse_img(img, units=3):
    # Cut image into units-by-units chunks
    H, W = img.shape[:2]
    dx = W//(units+1)
    dy = H//(units+1)

    windows = []
    for i in range(units):
        for j in range(units):
            t = dy*i
            b = dy*(i+2)
            l = dx*(j)
            r = dx*(j+2)

            windows.append((l, r, t, b))
    return windows

