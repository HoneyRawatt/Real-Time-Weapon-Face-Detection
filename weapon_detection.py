import cv2
import logging
from ultralytics import YOLO

from config import WEAPON_THRESHOLD

logger = logging.getLogger(__name__)

# Model and class list loaded once at module import — not per frame.
yolo_model = YOLO("best100.pt")


def _load_classes(path: str) -> list[str]:
    with open(path, "r") as fh:
        return [line.strip().lower() for line in fh if line.strip()]


classes = _load_classes("coco2.txt")


def detect_weapons(frame) -> bool:
    """
    Detect guns and knives in *frame* using the custom YOLO model.

    Draws red bounding boxes + confidence labels on *frame* in-place.
    Returns True if at least one weapon is found above WEAPON_THRESHOLD.
    """
    results = yolo_model(frame, verbose=False)
    weapon_detected = False

    for result in results:
        for idx in range(len(result.boxes)):
            conf = float(result.boxes.conf[idx])
            if conf < WEAPON_THRESHOLD:
                continue

            cls_idx    = int(result.boxes.cls[idx])
            class_name = classes[cls_idx]   # already lower-cased at load time

            if "gun" not in class_name and "knife" not in class_name:
                continue

            weapon_detected = True
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[idx])
            label = f"{class_name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA,
            )

    return weapon_detected
