import sys, argparse, copy
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
    image = 'cv2_tools/original.jpg'
    R = 53
    SNR = 5200

    # Read in image and prepare empty output image
    img_in = cv.imread(image, cv.IMREAD_GRAYSCALE)
    if img_in is None:
        sys.exit("ERROR : Image cannot be loaded...!!")

    ## [main]
    # it needs to process even image only
    roi = img_in[0:(img_in.shape[0] & -2), 0:(img_in.shape[1] & -2)]
    ## Hw calculation (start)
    h = calcPSF(roi.shape, R)
    Hw = calcWnrFilter(h, 1.0 / float(SNR))
    ## Hw calculation (stop)

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
    cv.imwrite("cv2_tools/result.jpg", imgOut)

## [help]
def help():
    print("2018-07-12")
    print("DeBlur_v8")
    print("You will learn how to recover an out-of-focus image by Wiener\
        filter")
## [help]

## [calcPSF]
def calcPSF(filterSize, R):
    h = np.zeros(filterSize, dtype=np.float32)
    point = (filterSize[1] // 2, filterSize[0] // 2)
    h = cv.circle(h, point, R, 255, -1, 8)
    summa = np.sum(h)
    return (h / summa) 
## [calcPSF]

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