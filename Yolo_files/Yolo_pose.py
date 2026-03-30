import cv2 as cv
from ultralytics import YOLO
import numpy as np

#Load the Yolo model
def load_model(model_path):
    model=YOLO(model_path)
    return model

def key_points_coordinate(results):
    keypoints_xy=[]
    for result in results: 
        if result.keypoints is not None:
            keypoints_xy = result.keypoints.xy #The coordinates of each keypoint 
            return keypoints_xy
        
def region_of_interest(img):
    height=img.shape[0]
    width=img.shape[1]
    polygon=np.array([[(width/2,0),(width,0),(width,height),(width/2,height)]],np.int32)
    mask=np.zeros_like(img)
    cv.fillPoly(mask,polygon,(255,255,255))
    return cv.bitwise_and(img,mask)
        
model=load_model("yolo11x-pose.pt")
img=cv.imread("Images/Penalty_1.png")
roi=region_of_interest(img)
results=model(roi)
result=results[0].plot() #Draw the keypoints and bounderies boxes in the image
coordinates=key_points_coordinate(results)
print(coordinates)
cv.imshow("pose",result)
cv.waitKey(0)

