import cv2
import cvzone
from ultralytics import YOLO
import math
from lib.sort import *

memory = {}
line = [(167,513), (546,591)]
counter = 0

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


# Read video file
cap = cv2.VideoCapture("data/video.mp4")


# download model if does not exist and the load the pretrained model
model = YOLO("weight_file/yolov8n.pt")

# Read mak image
image = cv2.imread("data/mask.png")

# Tracking
tracker = Sort(max_age=40,min_hits=3,iou_threshold=0.3)

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


  boxes = []
  indexIDs = []
  c = []
  previous = memory.copy()
  memory = {}
  
  tracker_results = tracker.update(detections)
  for trackers in tracker_results:
    X1,Y1,X2,Y2,ID = trackers
    width,height = X2 - X1,Y2 - Y1
    boxes.append([int(X1), int(Y1),int(width),int(height)])
    indexIDs.append(int(ID))
    memory[indexIDs[-1]] = boxes[-1]

  print(f"previous: {previous}")
  
  if len(boxes) > 0:
      i = 0
      for box in boxes:

        # Draw rectangle
        cvzone.cornerRect(img=frame,
                          bbox=[int(box[0]),int(box[1]),int(box[2]),int(box[3])],
                          l=7,
                          t=4)
        
        # Get the center of bouding box
        cx,cy = box[0]+(box[2]//2),(box[1]+box[3]//2)

        # Draw a circle on middle of the box
        cv2.circle(img=frame,center=(int(cx),int(cy)),radius=5,color=(255,0,255),thickness=-1)
        
        cv2.putText(img=frame,text=f"{indexIDs[i]}",org=(max(0,int(box[0])),max(25,int(box[1]))),
                    fontFace = cv2.FONT_HERSHEY_DUPLEX,
                    fontScale = 2.0,
                    color=(0, 255, 255),
                    thickness = 1
                    )
        i+=1
              
  # draw line
  cv2.line(frame, line[0], line[1], (0, 255, 255), 5)

  # draw counter
  cv2.putText(img= frame,                            # It is the image on which text is to be drawn.
              text= f"Vehical count: {counter}",     # Text string to be drawn.
              org = (46,44),                         # (X coordinate value, Y coordinate value).
              fontFace = cv2.FONT_HERSHEY_DUPLEX,    #It denotes the font type
              fontScale = 2.0,
              color=(0, 255, 255),
              thickness = 1                         # It is the thickness of the line in px.
              )

  # Display the resulting frame
  cv2.imshow("frame",frame)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()