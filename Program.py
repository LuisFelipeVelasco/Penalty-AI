import cv2 as cv
from ultralytics import YOLO
import numpy as np

x1_old, y1_old, x2_old, y2_old = 0, 0, 0, 0
frame =0
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
    #Hough Line algorithm to detect lines in the image
    detected_lines = cv.HoughLinesP(img_borders, 1, np.pi/180, 20, 
                                     minLineLength=120, maxLineGap=30)
    #The old coordinate values of the line to be used in case the new detected line is not valid
    global x1_old, y1_old, x2_old, y2_old 
    global frame
    #Set of lines with slope between 1 and 2 to be validated as goal line candidates
    validate_lines=[]
    #List of x1 values of the lines to be used to select the line with the smallest x1 value as the goal line candidate 
    x2_values=[]
    if detected_lines is not None:
        for line in detected_lines:
            x1, y1, x2, y2 = line[0] #The coordinates of the detected line
            m=(y2-y1)/(x2-x1) #The slope of the detected line
            if m>1 and m<2:
                validate_lines.append([x1,y1,x2,y2])
                x2_values.append(x2)
        if len(x2_values)!=0: #If there are lines with slope between 1 and 2, select the line with the smallest x1 value as the goal line candidate
            max_x2_value=min(x2_values)
            print(max_x2_value)
            index=x2_values.index(max_x2_value)
            goal_line=validate_lines[index]
            x1, y1, x2, y2 = goal_line #The coordinates of the goal line candidate
        else :
            x1, y1, x2, y2 = x1_old, y1_old, x2_old, y2_old #If there are no lines with slope between 1 and 2, use the old coordinates of the line

        #If the new detected line has a big difference in coordinates compared to the old line, it is not valid and the old line coordinates are used instead. This is to avoid sudden changes in the detected line due to noise or other factors.
        if ((abs(x1-x1_old)>15 or abs(y1-y1_old)>15 or abs(x2-x2_old)>15 or abs(y2-y2_old)>15) and x1_old!=0): #
            x1,y1,x2,y2=x1_old, y1_old, x2_old, y2_old
        #if the new  detected line has a small difference in coordinates compared to the old line, it is valid and the new line coordinates are used and stored as the old line coordinates for the next iteration
        elif (len(x2_values)!=0):
            x1_old, y1_old, x2_old, y2_old = goal_line
        #Draw the detected line on the image
        cv.line(img, (int(x1),int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)      
        return img
    
    else:
        #for line in detected_lines_memory:
        cv.line(img, (int(x1_old), int(y1_old)), (int(x2_old), int(y2_old)), (0, 0, 255), 2)
        return img
def put_text_on_image(img,text,b,g,r):
    height=img.shape[0]
    width=img.shape[1]
    cv.putText(img,text,(width//4,height-100),cv.FONT_HERSHEY_DUPLEX,2,(b,g,r),2,cv.LINE_AA)

Video=cv.VideoCapture("Videos/40.mp4")
while True:
    isTrue,Frame=Video.read()
    if isTrue:
        Roi=region_of_interest(Frame)
        White_video=detect_colors(Roi)
        Video_dilated=cv.dilate(White_video,(3,3),iterations=2)
        canny_video=cv.Canny(Video_dilated,20,50)
        Frame=lines_function(canny_video,Frame)
        put_text_on_image(Frame,"The goalkeeper is on line",0,0,255)
        cv.imshow("Video",Frame)
        if cv.waitKey(30) & 0xFF==ord('d'):
            break
    else:
        break