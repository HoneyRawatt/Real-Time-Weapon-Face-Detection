from ultralytics import YOLO


model = YOLO("yolov5su.pt")

def detect_people(frame):
    results = model(frame)[0]  
    person_boxes = []

    for box in results.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        if cls == 0 and conf > 0.5:  
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_boxes.append((x1, y1, x2, y2))

    return person_boxes
