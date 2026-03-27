import cv2 as cv
import numpy as np

def region_of_interest(img):
    height=img.shape[0]
    width=img.shape[1]
    polygon=np.array([[(5*width/8,5*height/8),(5*width/8,7*height/9),(7.5*width/10,7*height/9),(7.5*width/10,5*height/8)]],np.int32)
    mask=np.zeros_like(img)
    cv.fillPoly(mask,polygon,(255,255,255))
    return cv.bitwise_and(img,mask)

def detect_colors(img):
    img_hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
    white_lower = np.array([0, 10, 160]) 
    white_upper = np.array([180, 60, 200])
    mask=cv.inRange(img_hsv,white_lower,white_upper) #Mask to detect white color
    return mask 

def lines_function(img_borders, img):
    detected_lines = cv.HoughLinesP(img_borders, 1, np.pi/180, 40, 
                                     minLineLength=110, maxLineGap=20)   
    if detected_lines is not None:
        for line in detected_lines:
            x1, y1, x2, y2 = line[0]
            m=(y2-y1)/(x2-x1)
            if m>1 and m<2:
                print(x1, y1, x2, y2)
                cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        global detected_lines_memory
        detected_lines_memory = detected_lines
        print("New lines:")
        return img
    
    else:
        for line in detected_lines_memory:
            x1, y1, x2, y2 = line[0]
            m=(y2-y1)/(x2-x1)
            if m>1 and m<8: 
                cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return img

Video=cv.VideoCapture("Videos/02.mp4")
while True:
    isTrue,Frame=Video.read()
    if isTrue:
        Roi=region_of_interest(Frame)
        White_video=detect_colors(Roi)
        Video_dilated=cv.dilate(White_video,(3,3),iterations=2)
        canny_video=cv.Canny(Video_dilated,20,50)
        Frame=lines_function(canny_video,Frame)
        cv.imshow("Video",Frame)
        if cv.waitKey(30) & 0xFF==ord('d'):
            break
    else:
        break