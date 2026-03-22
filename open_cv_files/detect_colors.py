import cv2 as cv
import numpy as np

def detect_colors(img):
    img_hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV) #Become HSV to detect colors Hue, S Saturation, V Value
    # H (Hue): 0 to 180 (We don't care about the specific color tint, we include them all)
    # S (Saturation): 0 to 60 (Accepts everything from pure white to a slightly dirty/pale white)
    # V (Value/Brightness): 150 to 255 (Accepts from light gray caused by shadows, up to the brightest white)
    white_lower = np.array([0, 10, 160]) 
    white_upper = np.array([180, 60, 200])
    mask=cv.inRange(img_hsv,white_lower,white_upper) #Mask to detect white color
    return mask 

capture=cv.VideoCapture("Videos/06.mp4")
while True:
    isTrue,Frame=capture.read()
    if isTrue:
        mask=detect_colors(Frame)
        mask_dilated=cv.dilate(mask,(3,3),iterations=5)
        cv.imshow("Video",mask_dilated)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break
    else:
        break