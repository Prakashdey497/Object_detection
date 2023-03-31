import cv2
import cvzone
from ultralytics import YOLO
import math
from lib.sort import *


class VehicalUpDownCounting():
    """
    Vehical tracking and counting UP and Down
    """
    def __init__(self,video_path:str,model_path:str) -> None:
        super(VehicalUpDownCounting).__init__()

        self.up_counter = 0
        self.down_counter = 0
        self.video_path = video_path
        self.model_path = model_path

        # create a instance of tracker
        self.tracker = Sort(max_age=40,min_hits=3,iou_threshold=0.3)


    def main(self):
        
        # Read video file
        cap = cv2.VideoCapture(self.video_path)

        # download model if does not exist and the load the pretrained model
        model = YOLO("weight_file/yolov8n.pt")

        # Loop through over video frame
        while True:
            ret, frame = cap.read()

            if not ret: break

            results = model(frame)

            # create a empty array to store x1,y1,x2,y2 and confidence
            detections = np.empty((0,5))

            for result in results:
                for box in result.boxes:
        
                    # extract the class ID and confidence (i.e., probability) of the current object detection
                    classID = int(box.cls[0])
                    curent_class_names = result.names.get(classID)
                    confidence = float(box.conf[0])

                    if (curent_class_names == "car" or curent_class_names=="truck" or curent_class_names == "bus" or curent_class_names == "motorbike") and confidence > 0.5:

                        # Remove single dimension from bounding box array
                        squeeze_arr_box = np.squeeze(box.xyxy)

                        # Get the X coodinate ,Y coordinate ,Width and Height
                        X1,Y1,X2,Y2 = squeeze_arr_box

                        # Get width and height
                        width,height = int(X2 - X1),int(Y2 - Y1)

                        current_arr = np.array([int(X1),int(Y1),int(X2),int(Y2),confidence])
                        
                        detections = np.vstack(tup=(detections,current_arr))

    def intersect(self,A,B,C,D):
        return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)
    
    def ccw(self,A,B,C):
	    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    

obj = VehicalUpDownCounting(video_path="data/video.mp4")