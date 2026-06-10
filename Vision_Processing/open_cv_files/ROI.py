import cv2 as cv
import numpy as np
import os
BASE = os.path.dirname(os.path.abspath(__file__))

def region_of_interest(img):
    height=img.shape[0]
    width=img.shape[1]
    polygon=np.array([[(width/2,height/2),(width/2,height),(width,height),(width,height/2)]],np.int32)
    mask=np.zeros_like(img)
    cv.fillPoly(mask,polygon,(255,255,255))
    return cv.bitwise_and(img,mask)

capture=cv.VideoCapture(os.path.join(BASE, "../../Media/Videos/06.mp4"))
while True:
    isTrue,Frame=capture.read()
    if isTrue:
        Roi=region_of_interest(Frame)
        cv.imshow("Video",Roi)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break
    else:
        break