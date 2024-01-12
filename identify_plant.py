import cv2  
from ultralytics import YOLO
import math

# load model config and weights
net =  YOLO("yolo-Weights/yolov8n.pt")

# define classes
classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# load webcam
cap = cv2.VideoCapture(0)

# Function to Get Output Layers
while True:
    ret, img = cap.read()
    results = net(img, stream=True)
    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values


            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
             

            # class name
            cls = int(box.cls[0])
            if 0 <= cls < len(classes):
                if classes[cls] == "potted plant":

                     # put box in cam
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    print("Confidence --->", confidence)
                    print("Class name -->", classes[cls])
                    # object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (0, 0, 255)
                    thickness = 2

                    cv2.putText(img, classes[cls], org, font, fontScale, color, thickness)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (0, 0, 255)
                    thickness = 2

                    cv2.putText(img, "Not a plant", org, font, fontScale, color, thickness)
            else:
                print("Invalid class index:", cls)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
