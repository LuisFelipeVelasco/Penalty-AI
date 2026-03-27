from ultralytics import YOLO
import cv2 as cv

#Load the Yolo model
def load_model(model_path):
    model=YOLO(model_path)
    return model
#Call the function to load the model and store it in result variable
#Theres is multple suffix nano, small, medium, large, xlarge
model=load_model("yolov8n.pt")
#The model make the object detection in the image 
results=model("Images/Penalty_1.png")
#Inserts the bounding boxes and labels into the image
img = results[0].plot()
cv.imshow("Image", img)
cv.waitKey(0)
capture=cv.VideoCapture("Videos/06.mp4")
while True:
    isTrue,Frame=capture.read()
    if isTrue:
        results=model(Frame)
        result=results[0].plot()
        cv.imshow("Video",result)
        if cv.waitKey(10) & 0xFF==ord('d'):
            break
    else:
        break