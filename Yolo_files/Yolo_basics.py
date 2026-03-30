from sqlalchemy import true
from ultralytics import YOLO
import cv2 as cv
import numpy as np

#Load the Yolo model
def load_model(model_path):
    model=YOLO(model_path)
    return model

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

#Call the function to load the model and store it in result variable
#Theres is multple suffix nano, small, medium, large, xlarge
model=load_model("yolov8s.pt")
#The model make the object detection in the image 
results=model("Images/Penalty_1.png")
#Inserts the bounding boxes and labels into the image
img = results[0].plot()

capture=cv.VideoCapture("Videos/06.mp4")
last_coordinates=[]
global is_significant_change 
is_significant_change=False
while True:
    isTrue,Frame=capture.read()
    if isTrue:
        results=model(Frame)
        for box in results[0].boxes: #boxes is used to get the bounding box coordinates
            if box.cls==32: #If the class of the box is 32 (which corresponds to a soccer ball in the COCO dataset)
                x1,y1,x2,y2=box.xyxy[0].tolist() #We use list unpacking to get the coordinates of the bounding box of the soccer ball
                print(x1,y1,x2,y2)
                if last_coordinates: 
                    is_significant_change = evaluate_change(x1, x2, y1, y2, last_coordinates) 
                last_coordinates=[x1,y1,x2,y2]
        result=results[0].plot()
        cv.imshow("Video",result)
        if is_significant_change: #if the position of the soccer ball change show that frame with the bounding boxes and stop the video                         
            cv.imshow("Video", result)
            cv.waitKey(0)
            break
        
        if cv.waitKey(1) & 0xFF==ord('d'):
            break
    else:
        break