import cv2 as cv
#img=cv.imread("Images/Chimpanzini_Bananini.jpg")
#cv.imshow("Chimpanzini_Bananini",img)
#cv.waitKey(0)

capture=cv.VideoCapture(0)

def rescaleFrame(frame,scale=0.35):
    width=int(frame.shape[1]*scale)
    height=int(frame.shape[0]*scale)
    dimensions=(width,height)
    return cv.resize(frame,dimensions,interpolation=cv.INTER_AREA)

while True:
    isTrue,frame=capture.read()
    if isTrue:
        video_resized=rescaleFrame(frame)
        cv.imshow("Video Resized",video_resized)
        if cv.waitKey(10) & 0xFF==ord('d'):
            break
    else:
        break