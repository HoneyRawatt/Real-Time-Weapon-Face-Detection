import logging
from ultralytics import YOLO

from config import PERSON_THRESHOLD

logger = logging.getLogger(__name__)

# YOLOv8n pretrained on COCO — lighter and faster than YOLOv5su for this task.
# yolov8n.pt is already present in the repo root.
model = YOLO("yolov8n.pt")


def detect_people(frame) -> list[tuple[int, int, int, int]]:
    """
    Detect persons (COCO class 0) in *frame*.

    Returns a list of (x1, y1, x2, y2) integer bounding boxes for every
    detection that exceeds PERSON_THRESHOLD confidence.
    Does NOT draw annotations — that is handled by the pipeline so the
    drawing colour is consistent regardless of detection module.
    """
    results = model(frame, verbose=False)[0]
    person_boxes: list[tuple[int, int, int, int]] = []

    for box in results.boxes:
        if int(box.cls.item()) == 0 and float(box.conf.item()) >= PERSON_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_boxes.append((x1, y1, x2, y2))

    return person_boxes
