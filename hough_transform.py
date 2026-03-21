import cv2 as cv
import numpy as np

def lines(img_canny,img):
    lines = cv.HoughLinesP(img_canny,1,np.pi/180,50,minLineLength=300,maxLineGap=25)
    for line in lines:
        x1,y1,x2,y2=line[0]
        cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    return img
Capture=cv.VideoCapture("Videos/07.mp4")

while True:
    isTrue,Frame=Capture.read()
    frame_number = int(Capture.get(cv.CAP_PROP_POS_FRAMES))
    print(frame_number)
    #if frame_number==1:
        #ret, first_frame = Capture.read()
    ret,frame=Capture.read()
    if isTrue:
        img_blur=cv.GaussianBlur(Frame,(15,15),0)
        img_gray=cv.cvtColor(img_blur,cv.COLOR_BGR2GRAY)
        img_canny=cv.Canny(img_gray,10,120)
        img_dilated=cv.dilate(img_canny,(7,7),iterations=5)
        Frame=lines(img_dilated,Frame)
        cv.imshow("Video",Frame)
        if cv.waitKey(30) & 0xFF==ord('d'):
            break
    else:
        break