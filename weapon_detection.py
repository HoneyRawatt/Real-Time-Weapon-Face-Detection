import cv2
from ultralytics import YOLO


yolo_model = YOLO('best100.pt')


def load_classes_from_file(file_path):
    with open(file_path, 'r') as f:
        classes = f.read().strip().split('\n')
    return classes

classes = load_classes_from_file('coco2.txt')


def detect_weapons(frame):
    results = yolo_model(frame)
    weapon_detected = False

    
    for result in results:
        cls = result.boxes.cls
        conf = result.boxes.conf
        detections = result.boxes.xyxy

        for pos, detection in enumerate(detections):
            if conf[pos] >= 0.6:
                xmin, ymin, xmax, ymax = detection
                label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"

                
                if 'gun' in label or 'knife' in label:
                    weapon_detected = True
                    
                    
                    color = (0, 0, 255) 
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)  

                   
                    cv2.putText(frame, label, (int(xmin), int(ymin) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)  

    return weapon_detected
