import cv2 as cv
import os
BASE = os.path.dirname(os.path.abspath(__file__))

img=cv.imread(os.path.join(BASE, "../../Media/Images/Penalty_1.png"))
cv.imshow("Penalty_1",img)
cv.waitKey(0)
img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#cv.imshow("Penalty_1_Gray",img_gray)

capture=cv.VideoCapture(os.path.join(BASE, "../../Media/Videos/01.mp4"))
while True:
    isTrue,frame=capture.read()
    if isTrue:
        blur_frame=cv.GaussianBlur(frame,(21,21),0)
        gray_frame=cv.cvtColor(blur_frame,cv.COLOR_BGR2GRAY)
        Canny_frame=cv.Canny(gray_frame,10,15)
        cv.imshow("Video",Canny_frame)
        if cv.waitKey(20) & 0xFF==ord('d'):
            break