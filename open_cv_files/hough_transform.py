import cv2 as cv
import numpy as np

def lines_function(img_canny, img):
    global last_lines
    # Subí el maxLineGap a 100 para que salte el pie del arquero
    detected_lines = cv.HoughLinesP(img_canny, 1, np.pi/180, 80, 
                                     minLineLength=120, maxLineGap=25)
    
    # SI DETECTÓ LÍNEAS: las dibujamos y actualizamos la memoria
    if detected_lines is not None:
        last_lines = detected_lines # Guardamos en memoria
        for line in detected_lines:
            x1, y1, x2, y2 = line[0]
            cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return img

Capture=cv.VideoCapture("Videos/07.mp4")

while True:
    isTrue,Frame=Capture.read()
    frame_number = int(Capture.get(cv.CAP_PROP_POS_FRAMES))
    print(frame_number)
    #if frame_number==1:
        #ret, first_frame = Capture.read()
    if isTrue:
        img_blur=cv.GaussianBlur(Frame,(5,5),100)
        img_gray=cv.cvtColor(img_blur,cv.COLOR_BGR2GRAY)
        img_canny=cv.Canny(img_gray,150,200)
        img_dilated=cv.dilate(img_canny,(3,3),iterations=2)
        img_eroded=cv.erode(img_dilated,(3,3),iterations=1)
        Frame=lines_function(img_eroded,Frame)
        cv.imshow("Video",Frame)
        if cv.waitKey(30) & 0xFF==ord('d'):
            break
    else:
        break