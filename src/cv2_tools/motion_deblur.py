import sys, argparse, copy
from cv2 import FILLED
import numpy as np
import cv2 as cv

def main():
    # Parse arguments
    # text = "Recover an out-of-focus image by Wiener filter."
    # parser = argparse.ArgumentParser(text)
    # parser.add_argument("--image", type=str, required=True, 
    # 	help="Specify the input image filename.")
    # parser.add_argument("--R", type=int, required=True,
    # 	help="Specify the point spread circle radius. Demo example: 53")
    # parser.add_argument("--SNR", type=float, required=True,
    # 	help="Specify the signal-to-noise ratio (SNR). Demo example: 5200")
    # args = parser.parse_args()

    help()
    image = 'cv2_tools/original_motion.JPG'
    len = 125
    theta = 0
    SNR = 700
    gamma = 5.0
    beta = 0.2

    # Read in image and prepare empty output image
    img_in = cv.imread(image, cv.IMREAD_GRAYSCALE)
    if img_in is None:
        sys.exit("ERROR : Image cannot be loaded...!!")

    ## [main]
    # it needs to process even image only
    roi = img_in[0:(img_in.shape[0] & -2), 0:(img_in.shape[1] & -2)]
    ## Hw calculation (start)
    h = calcPSF(roi.shape, len, theta)
    Hw = calcWnrFilter(h, 1.0 / float(SNR))
    ## Hw calculation (stop)

    img_in = img_in.astype(np.float32)
    img_in = edgetaper(img_in, gamma, beta)

    ## filtering (start)
    imgOut = filter2DFreq(roi, Hw)
    ## filtering (stop)
    ## [main] 

    imgOut = imgOut.astype(np.float32)
    imgOut = cv.normalize(imgOut, imgOut, alpha=0, beta=255, 
        norm_type=cv.NORM_MINMAX)
    # cv.startWindowThread()
    # cv.namedWindow("photo")
    # cv.imshow('photo', imgOut)
    # cv.waitKey()
    cv.imwrite("cv2_tools/result_motion.jpg", imgOut)

## [help]
def help():
    print("2018-07-12")
    print("DeBlur_v8")
    print("You will learn how to recover an out-of-focus image by Wiener\
        filter")
## [help]

## [calcPSF]
def calcPSF(filterSize, len, theta):
    h = np.zeros(filterSize, dtype=np.float32)
    point = (filterSize[1] // 2, filterSize[0] // 2)
    h = cv.ellipse(h, point, (0, np.int32(len/2.0)), 90.0 - theta, 0, 360, 255)
    summa = np.sum(h)
    return (h / summa) 
## [calcPSF]

## [edgetaper]
def edgetaper(inputImg, gamma, beta):
    Ny, Nx = inputImg.shape
    p1 = np.zeros((1, Nx))
    p2 = np.zeros((Ny, 1))

    dx = 2*np.pi/Nx
    x = -np.pi
    for i in np.arange(0, Nx):
        p1[0,i] = 0.5*(np.tanh(((x + gamma)/2)/beta) - np.tanh(((x - gamma)/2)/beta))
        x = x + dx

    dy = 2*np.pi/Ny
    y = -np.pi
    for i in np.arange(0, Ny):
        p2[i] = 0.5*(np.tanh(((y + gamma)/2)/beta) - np.tanh(((y - gamma)/2)/beta))
        y = y + dx

    w = p2 @ p1
    w = w.astype(np.float32)
    return cv.multiply(inputImg, w)
## [edgetaper]

## [filter2DFreq]
def filter2DFreq(inputImg, H):
    planes = [inputImg.copy().astype(np.float32), 
        np.zeros(inputImg.shape, dtype=np.float32)]
    complexI = cv.merge(planes)
    complexI = cv.dft(complexI, flags=cv.DFT_SCALE)

    planesH = [H.copy().astype(np.float32), 
        np.zeros(H.shape, dtype=np.float32)]
    complexH = cv.merge(planesH)
    complexIH = cv.mulSpectrums(complexI, complexH, 0)

    complexIH = cv.idft(complexIH)
    planes = cv.split(complexIH)
    return planes[0]
## [filter2DFreq]

## [calcWnrFilter]
def calcWnrFilter(input_h_PSF, nsr):
    h_PSF_shifted = np.fft.fftshift(input_h_PSF)
    planes = [h_PSF_shifted.copy().astype(np.float32), 
        np.zeros(h_PSF_shifted.shape, dtype=np.float32)]
    complexI = cv.merge(planes)
    complexI = cv.dft(complexI)
    planes = cv.split(complexI)
    denom = np.power(np.abs(planes[0]), 2)
    denom += nsr 
    return cv.divide(planes[0], denom) 
## [calcWnrFilter]

if __name__ == "__main__":
    main()
    cv.destroyAllWindows()