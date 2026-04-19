"""
Shared detection pipeline.

Both the CLI entry-point (main.py) and the Flask web server (app.py) use
the functions defined here so that detection logic lives in exactly one
place.  Previously the ~120-line detection loop was copy-pasted between
detect_objects_in_realtime() and generate_frames().

Public API
----------
DetectionState          — mutable alarm/email bookkeeping per camera session
process_frame()         — run all three detectors on one frame; handle alarm/email
detect_objects_in_realtime()  — CLI camera loop
generate_frames()             — MJPEG generator for Flask streaming
"""
import cv2
import threading
import time
import logging
from dataclasses import dataclass, field

from face_detection   import detect_faces
from weapon_detection import detect_weapons
from person_detection import detect_people
from alarm            import start_alarm, stop_alarm
from utils            import load_known_faces
from config           import (
    CAMERA_INDEX,
    FRAME_SKIP,
    FACE_THRESHOLD,
    ALARM_COOLDOWN,
    ALARM_EMAIL_THRESHOLD,
    EMBEDDINGS_FILE,
)

logger = logging.getLogger(__name__)


# ── State ────────────────────────────────────────────────────────────────────

@dataclass
class DetectionState:
    """Per-session alarm and email bookkeeping.  One instance per camera loop."""
    alarm_playing:  bool  = False
    last_alarm_time: float = field(default_factory=float)
    alarm_count:    int   = 0


# ── Core detection pass ───────────────────────────────────────────────────────

def process_frame(
    frame,
    known_faces: dict,
    state: DetectionState,
) -> tuple[bool, bool, bool]:
    """
    Run all three detectors on a single *frame*.

    Face and weapon detectors run in parallel background threads while the
    person detector runs on the calling thread, so the wall-clock latency
    is roughly max(face_time, weapon_time) + person_time instead of the sum.

    Both detector threads write directly to *frame* (bounding boxes / labels).
    A threading.Lock is used to protect the shared results dict; the GIL
    alone is not sufficient when OpenCV releases it in C++ code.

    Side-effects:
        • Triggers / stops audio alarm via alarm.py
        • Fires a background email thread after ALARM_EMAIL_THRESHOLD alarms
        • Resets alarm_count after each email so the alert rate stays bounded

    Returns:
        (person_detected, unknown_detected, weapon_detected) — booleans
    """
    result: dict = {}
    lock = threading.Lock()

    def _run_face():
        val = detect_faces(frame, known_faces, FACE_THRESHOLD)
        with lock:
            result["face"] = val

    def _run_weapon():
        val = detect_weapons(frame)
        with lock:
            result["weapon"] = val

    face_t   = threading.Thread(target=_run_face,   daemon=True)
    weapon_t = threading.Thread(target=_run_weapon, daemon=True)
    face_t.start()
    weapon_t.start()

    # Person detection runs on the calling thread while the other two are busy.
    person_boxes    = detect_people(frame)
    person_detected = len(person_boxes) > 0

    face_t.join()
    weapon_t.join()

    unknown_detected = result.get("face",   False)
    weapon_detected  = result.get("weapon", False)

    # Draw person boxes (face/weapon boxes drawn by their own modules).
    for (x1, y1, x2, y2) in person_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            frame, "Person", (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2,
        )

    logger.debug(
        f"Person:{person_detected}  Unknown:{unknown_detected}  Weapon:{weapon_detected}"
    )

    # ── Alarm / email logic ───────────────────────────────────────────────────
    now    = time.time()
    threat = person_detected and unknown_detected and weapon_detected

    if threat:
        if not state.alarm_playing or (now - state.last_alarm_time > ALARM_COOLDOWN):
            logger.warning("THREAT DETECTED — person + unknown face + weapon.")
            threading.Thread(target=start_alarm, daemon=True).start()
            state.alarm_playing   = True
            state.last_alarm_time = now
            state.alarm_count    += 1

            if state.alarm_count >= ALARM_EMAIL_THRESHOLD:
                from email_sender import send_email_with_attachment
                snapshot = frame.copy()

                def _send(img):
                    path = "screenshot.png"
                    cv2.imwrite(path, img)
                    send_email_with_attachment(path)

                threading.Thread(target=_send, args=(snapshot,), daemon=True).start()
                state.alarm_count = 0       # reset — prevents email every frame

    elif state.alarm_playing:
        logger.info("Threat condition cleared — stopping alarm.")
        stop_alarm()
        state.alarm_playing = False

    return person_detected, unknown_detected, weapon_detected


# ── Camera loops ──────────────────────────────────────────────────────────────

def _open_camera() -> cv2.VideoCapture:
    """Open camera at CAMERA_INDEX; logs a clear error if unavailable."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        logger.error(
            f"Cannot open camera at index {CAMERA_INDEX}.  "
            "Check the camera is connected and set CAMERA_INDEX in .env "
            "to the correct device number (0 is usually the built-in webcam)."
        )
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def detect_objects_in_realtime() -> None:
    """
    CLI detection loop.

    Opens the configured camera, runs the detection pipeline, and displays
    the annotated feed in a cv2 window.  Press 'q' to quit.
    """
    cap = _open_camera()
    if not cap.isOpened():
        return

    known_faces      = load_known_faces(EMBEDDINGS_FILE)
    state            = DetectionState()
    frame_count      = 0
    last_reload_time = time.time()

    logger.info(
        f"CLI detection started — camera {CAMERA_INDEX}, "
        f"frame-skip {FRAME_SKIP}.  Press 'q' to quit."
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Camera read failed — stopping.")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        # Reload enrolled faces every 30 s so new enrolments are picked up
        # without restarting the process.
        now = time.time()
        if now - last_reload_time >= 30:
            known_faces      = load_known_faces(EMBEDDINGS_FILE)
            last_reload_time = now

        process_frame(frame, known_faces, state)
        cv2.imshow("SafeGuard — Real-time Detection  (q = quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()
    logger.info("CLI detection stopped.")


def generate_frames(stop_event: threading.Event, streaming_flag):
    """
    MJPEG frame generator for Flask streaming.

    Yields multipart JPEG byte chunks suitable for a
    ``multipart/x-mixed-replace`` HTTP response.

    Args:
        stop_event:     threading.Event — set by /stop_stream to end the loop.
        streaming_flag: callable () → bool — returns False to end the loop.
    """
    cap = _open_camera()
    if not cap.isOpened():
        return

    known_faces      = load_known_faces(EMBEDDINGS_FILE)
    state            = DetectionState()
    frame_count      = 0
    last_reload_time = time.time()

    logger.info("Web stream started.")

    while streaming_flag() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            logger.error("Camera read failed in web stream.")
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        # Reload enrolled faces every 30 s so new enrolments are picked up
        # without restarting the stream.
        now = time.time()
        if now - last_reload_time >= 30:
            known_faces      = load_known_faces(EMBEDDINGS_FILE)
            last_reload_time = now

        process_frame(frame, known_faces, state)

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

    cap.release()
    stop_alarm()
    logger.info("Web stream stopped.")
