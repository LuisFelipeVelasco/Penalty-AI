import cv2 as cv
import numpy as np          

def rescaleFrame(Frame,scale=0.30):
    width=int(Frame.shape[1]*scale)
    height=int(Frame.shape[0]*scale)
    dimensions=(width,height)
    return cv.resize(Frame,dimensions,interpolation=cv.INTER_AREA)

Capture=cv.VideoCapture("Videos/VID_20260206_083910.mp4")

while True:
    isTrue,Frame=Capture.read()
    if isTrue:
        cv.putText(Frame,"Is in line?",(80,200),cv.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        Frame=rescaleFrame(Frame)
        cv.imshow("Video",Frame)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break
    else:
        break
