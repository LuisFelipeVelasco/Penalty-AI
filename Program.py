import cv2 as cv
from ultralytics import YOLO
import numpy as np

x1_old, y1_old, x2_old, y2_old = 0, 0, 0, 0
frame =0
def region_of_interest(img,x0,y0,x1,y1):
    height=img.shape[0]
    width=img.shape[1]
    polygon=np.array([[(x0,y0),(x0,y1),(x1,y1),(x1,y0)]],np.int32)
    mask=np.zeros_like(img)
    cv.fillPoly(mask,polygon,(255,255,255))
    return cv.bitwise_and(img,mask)

def detect_colors(img):
    img_hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)
    white_lower = np.array([0, 10, 160]) 
    white_upper = np.array([180, 60, 200])
    mask=cv.inRange(img_hsv,white_lower,white_upper) #Mask to detect white color
    return mask 

def lines_function(img_borders):
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
        #cv.line(img, (int(x1),int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)      
        return [x1, y1, x2, y2]
    
    else:
        #for line in detected_lines_memory:
        #cv.line(img, (int(x1_old), int(y1_old)), (int(x2_old), int(y2_old)), (0, 0, 255), 2)
        return [x1_old, y1_old, x2_old, y2_old]
    
def load_model(model_path):
    model=YOLO(model_path)
    return model

def Foot_points_coordinate(results):
    keypoints_xy=[]
    for result in results: 
        if result.keypoints is not None:
            keypoints_xy = result.keypoints.xy #The coordinates of each keypoint
            lenght_keypoints=len(keypoints_xy[0]) #The number of keypoints detected in the image
            leg_foot_x=keypoints_xy[0][lenght_keypoints-2][0] #The coordinate x of the first foot keypoint
            leg_foot_y=keypoints_xy[0][lenght_keypoints-2][1] #The coordinate y of the first foot keypoint
            right_foot_x=keypoints_xy[0][lenght_keypoints-1][0] #The coordinate x of the second foot keypoint
            right_foot_y=keypoints_xy[0][lenght_keypoints-1][1] #The coordinate y of the second foot keypoint
            return [int(leg_foot_x), int(leg_foot_y+25), int(right_foot_x), int(right_foot_y+20)]

def put_text_on_image(img,text,b,g,r):
    height=img.shape[0]
    width=img.shape[1]
    cv.putText(img,text,(width//4,height-100),cv.FONT_HERSHEY_DUPLEX,2,(b,g,r),2,cv.LINE_AA)

def is_on_the_goal_line(foot_coordinates,line_coordinates):
    foot_x1, foot_y1, foot_x2, foot_y2 = foot_coordinates
    line_x1, line_y1, line_x2, line_y2 = line_coordinates
    m=(line_y2-line_y1)/(line_x2-line_x1) #The slope of the detected line
    Y1=m*(foot_x1-line_x1)+line_y1 #The y coordinate of the point on the line with the same x coordinate as the first foot keypoint
    Y2=m*(foot_x2-line_x1)+line_y1 #The y coordinate of the point on the line with the same x coordinate as the second foot keypoint
    if (foot_y1<Y1 or foot_y2<Y2):
        return True
    else:
        return False

#evaluate_change look if the coordinates of the object change more than 5 , in that case returns true
def evaluate_change(x1,x2,y1,y2,last_coordinates):
    last_x1, last_y1, last_x2, last_y2 = last_coordinates
    change_x1 = abs(x1 - last_x1)
    change_x2 = abs(x2 - last_x2)
    change_y1= abs(y1 - last_y1)
    change_y2=abs(y2 - last_y2) 
    if (change_x1>5 and change_x2 >5) or (change_y1>5 and change_y2 >5) :
        return True
    else:
        return False

Video=cv.VideoCapture("Videos/14.mp4")
last_coordinates=[]
global is_significant_change 
is_significant_change=False
model1=load_model("yolo11l-pose.pt")
model2=load_model("yolov8m.pt")

while True:
    isTrue,Frame=Video.read()
    if isTrue:      
        height=Frame.shape[0]
        width=Frame.shape[1]
        
        #Detect position foot keypoints using Yolo pose estimation model
        roi_detect_foot=region_of_interest(Frame,width/2,0,width, height)
        results=model1(roi_detect_foot)
        Foot_points_coordinate(results)
        coordinates=Foot_points_coordinate(results)

        #Detect position of the goal line using Hough Line algorithm
        roi_detect_line=region_of_interest(Frame,5*width/8,5*height/8,7.5*width/10,7*height/9)
        White_video=detect_colors(roi_detect_line)
        Video_dilated=cv.dilate(White_video,(3,3),iterations=2)
        canny_video=cv.Canny(Video_dilated,20,50)
        Line_Frame=lines_function(canny_video)

        #Detect if the goalkeeper is on the line by comparing the position of the foot keypoints with the position of the line
        if (is_on_the_goal_line(coordinates,Line_Frame)):
             put_text_on_image(Frame,"The goalkeeper is on line",0,255,0)
        else:
             put_text_on_image(Frame,"The goalkeeper is not on line",0,0,255)

        #Detect when the shooter shoots the ball
        results=model2(Frame)
        result=results[0].plot()
        for box in results[0].boxes: #boxes is used to get the bounding box coordinates
            if box.cls==32: #If the class of the box is 32 (which corresponds to a soccer ball in the COCO dataset)
                x1,y1,x2,y2=box.xyxy[0].tolist() #We use list unpacking to get the coordinates of the bounding box of the soccer ball
                if last_coordinates: 
                    is_significant_change = evaluate_change(x1, x2, y1, y2, last_coordinates)
                last_coordinates=[x1,y1,x2,y2]
        result=results[0].plot()
        x1_line, y1_line, x2_line, y2_line = Line_Frame
        cv.line(result, (int(x1_line),int(y1_line)), (int(x2_line), int(y2_line)), (0, 0, 255), 2) #Draw the detected line on the image                         
        cv.imshow("Video",result)
        if is_significant_change: #if the position of the soccer ball change show that frame with the bounding boxes and stop the video
            
            cv.imshow("Video", result)
            cv.waitKey(0)
            break
        if cv.waitKey(1) & 0xFF==ord('d'):
            break
    else:
        break