import cv2
from ultralyticsplus import YOLO
from ultralytics import YOLO
import cv2
import random
import math
from collections import Counter

# inference
image_path = "static/img/img_normal.jpg"

classNames = ['Blast', 'Blight', "Brown Spot", "Healthy", "Tungro"]

def video_detection(path_x):
    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))

    model=YOLO("best.pt")
    while True:
        success, img = cap.read()
        results=model(img,stream=True)
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                print(x1,y1,x2,y2)
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                print(t_size)
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
        yield img
cv2.destroyAllWindows()

def detection_yolo():
    title = "YOLOv8"
    # load model
    model = YOLO("best.pt")
    # set model parameters
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image
    
    array = []
    counts = Counter()
    width = int(640)
    height = int(640)
    dim = (width, height)
    img = cv2.imread(image_path)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    results = model(resized, show = False)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #print(x1, y1, x2, y2)

            color1 = random.randrange(128, 255)
            color2 = random.randrange(128, 255)
            color3 = random.randrange(128, 255)

            # Draw filled rectangle as background for text
            cv2.rectangle(resized, (max(0, x1), max(35, y1) - 25), (x2, y1), (color1, color2, color3), -1)
            
            # Draw bounding box
            cv2.rectangle(resized,(x1,y1),(x2,y2),(color1,color2, color3),3)

            conf = math.ceil((box.conf[0]*100))/100

            cls = int(box.cls[0])

            cv2.putText(img=resized, text=f'{classNames[cls]} {conf}', org=(max(0, x1), max(35, y1) - 5),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 0, 0), thickness=2)
            array.append(classNames[cls])
            counts[classNames[cls]] += 1
    print(array)
    
    for label, count in counts.items():
        print(f'{label} = {count}')
    
    cv2.imwrite("static/img/img_now.jpg", resized)
    return counts, title