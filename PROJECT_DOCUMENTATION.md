# SafeGuard — Complete Project Documentation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [How the System Works — High Level](#2-how-the-system-works--high-level)
3. [Project Structure](#3-project-structure)
4. [Architecture & Module Ownership](#4-architecture--module-ownership)
5. [Module Reference](#5-module-reference)
   - [config.py](#51-configpy)
   - [pipeline.py](#52-pipelinepy)
   - [face_detection.py](#53-face_detectionpy)
   - [weapon_detection.py](#54-weapon_detectionpy)
   - [person_detection.py](#55-person_detectionpy)
   - [app.py](#56-apppy)
   - [main.py](#57-mainpy)
   - [utils.py](#58-utilspy)
   - [alarm.py](#59-alarmpy)
   - [email_sender.py](#510-email_senderpy)
6. [Detection Pipeline — Deep Dive](#6-detection-pipeline--deep-dive)
7. [Face Recognition Pipeline — Deep Dive](#7-face-recognition-pipeline--deep-dive)
8. [Weapon Detection Pipeline — Deep Dive](#8-weapon-detection-pipeline--deep-dive)
9. [Person Detection Pipeline — Deep Dive](#9-person-detection-pipeline--deep-dive)
10. [Alarm & Alert System](#10-alarm--alert-system)
11. [Web Interface](#11-web-interface)
12. [Data Flow Diagrams](#12-data-flow-diagrams)
13. [Threading Model](#13-threading-model)
14. [Configuration Reference](#14-configuration-reference)
15. [Models Reference](#15-models-reference)
16. [Environment Setup & Installation](#16-environment-setup--installation)
17. [Running the System](#17-running-the-system)
18. [Enrolling Safe People](#18-enrolling-safe-people)
19. [Performance Characteristics](#19-performance-characteristics)
20. [Security Design](#20-security-design)
21. [Technology Decisions](#21-technology-decisions)
22. [Troubleshooting Guide](#22-troubleshooting-guide)
23. [Project Evolution & Upgrade History](#23-project-evolution--upgrade-history)

---

## 1. Project Overview

**SafeGuard** is a real-time computer vision security surveillance system built entirely in Python. It monitors a live camera feed and detects potential threats by running three independent AI models simultaneously on every frame.

### The core threat condition

The system raises an alarm **only** when all three conditions are true at the same time:

```
Person present  AND  Unknown face (not in safe list)  AND  Weapon visible
```

This three-way AND condition dramatically reduces false alarms. A person with a gun who is enrolled in the safe list does not trigger an alarm. An unknown face without a weapon does not trigger an alarm. All three must be detected simultaneously.

### What happens when a threat is detected

1. An audio alarm plays immediately through the system speakers
2. The alarm counter increments
3. After the alarm fires `ALARM_EMAIL_THRESHOLD` times (default 3), a screenshot is saved and emailed to the configured recipient with the frame as an attachment
4. The alarm counter resets to zero after each email (prevents continuous spam)
5. When the threat condition clears, the alarm stops automatically

### Two run modes

| Mode | Entry point | Display |
|------|-------------|---------|
| **Web UI** | `python app.py` | Browser at `http://127.0.0.1:5000` |
| **CLI** | `python main.py` | OpenCV window on desktop |

Both modes use the exact same detection logic from `pipeline.py` — there is no code duplication between them.

---

## 2. How the System Works — High Level

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CAMERA FEED                                  │
│                     (640×480, up to 30 fps)                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │  raw BGR frames
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FRAME SKIP GATE                                 │
│              Process every 3rd frame (configurable)                  │
│              Reduces CPU load ~3× with minimal accuracy loss         │
└────────────────────────────┬────────────────────────────────────────┘
                             │  selected frames only
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   PARALLEL DETECTION LAYER                           │
│                                                                      │
│  Thread 1: face_detection.py          Thread 2: weapon_detection.py │
│  ┌────────────────────────────┐       ┌──────────────────────────┐  │
│  │ InsightFace SCRFD-500M     │       │ Custom YOLO11s           │  │
│  │ → find all faces           │       │ (best100.pt)             │  │
│  │                            │       │ → find guns, knives      │  │
│  │ InsightFace ArcFace        │       │                          │  │
│  │ w600k_mbf (512-d)          │       │ Filter: conf ≥ 0.60      │  │
│  │ → embed each face          │       │ Class: "gun" or "knife"  │  │
│  │                            │       │                          │  │
│  │ Cosine similarity vs       │       │ Draw RED box on frame    │  │
│  │ enrolled known_faces dict  │       │                          │  │
│  │                            │       │ Returns: weapon_detected │  │
│  │ GREEN box = known person   │       └──────────────────────────┘  │
│  │ RED box   = unknown person │                                      │
│  │ Returns: unknown_detected  │       person_detection.py           │
│  └────────────────────────────┘       ┌──────────────────────────┐  │
│                                       │ YOLOv8n (COCO pretrained) │  │
│                 (calling thread)      │ → find persons            │  │
│                                       │ Filter: class=0, conf≥0.5 │  │
│                                       │ Returns: list of boxes    │  │
│                                       └──────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │  (person_detected, unknown_detected, weapon_detected)
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      THREAT ANALYSIS                                 │
│                                                                      │
│   IF person_detected AND unknown_detected AND weapon_detected:       │
│       → play alarm (pygame)                                          │
│       → increment alarm_count                                        │
│       → IF alarm_count >= 3: save screenshot → send email           │
│       → reset alarm_count to 0 after email                          │
│   ELSE IF alarm was playing:                                         │
│       → stop alarm                                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │  annotated frame (boxes + labels drawn in-place)
                             ▼
                    ┌────────┴────────┐
                    │                 │
              Flask MJPEG          cv2.imshow
              (Web UI)             (CLI mode)
```

---

## 3. Project Structure

```
Real-Time-Weapon-Face-Detection/
│
├── Core application
│   ├── app.py                  Flask web server — routes, enrollment, streaming
│   ├── main.py                 CLI entry point (one-liner — calls pipeline)
│   ├── pipeline.py             SINGLE SOURCE OF TRUTH for all detection logic
│   ├── config.py               All constants, backed by environment variables
│   └── utils.py                Embedding file loader with validation
│
├── Detector modules
│   ├── face_detection.py       InsightFace SCRFD + ArcFace face recognition
│   ├── weapon_detection.py     Custom YOLO11s weapon detector
│   └── person_detection.py     YOLOv8n person detector
│
├── Alert modules
│   ├── alarm.py                pygame audio alarm (lazy init)
│   └── email_sender.py         Gmail SMTP alert with screenshot attachment
│
├── Models
│   ├── best100.pt              Custom-trained YOLO11s (100 epochs, weapon classes)
│   ├── yolov8n.pt              YOLOv8n pretrained on COCO (person detection)
│   └── coco2.txt               Class label file for best100.pt
│
├── Audio
│   └── alarm.wav               Alarm sound file (played by pygame)
│
├── Face data  (git-ignored — stays local)
│   ├── facenet_embeddings1.npy Enrolled ArcFace embeddings {name: 512-d vector}
│   └── dataset/                Raw face images, one subfolder per person
│       └── PersonName/
│           ├── photo1.jpg
│           └── photo2.jpg
│
├── Web UI
│   ├── templates/
│   │   ├── index.html          Home page with navigation cards
│   │   ├── detection.html      Live stream viewer (Start/Stop buttons)
│   │   ├── safepeople.html     Table of enrolled people
│   │   └── create_encoding.html  Add / delete people form
│   └── static/
│       └── styles.css          Home page custom CSS (glowing dot background)
│
├── Developer scripts
│   ├── scripts/check_env.py        Print loaded env vars (masks password)
│   └── scripts/send_test_email.py  Send a one-off test alert email
│
├── Configuration
│   ├── .env                    Your credentials (git-ignored — never commit)
│   ├── .env.example            Template showing which vars to set
│   ├── requirements.txt        Python package dependencies
│   └── .gitignore
│
└── Documentation
    ├── README.md               Quick-start guide
    └── PROJECT_DOCUMENTATION.md   This file
```

---

## 4. Architecture & Module Ownership

### Dependency graph

```
main.py ──────────────────────────────────────────────────┐
                                                           │
app.py ────── pipeline.py ──── face_detection.py ─── config.py
               │           │── weapon_detection.py ─┘    │
               │           │── person_detection.py        │
               │           │── alarm.py                   │
               │           │── utils.py ─────────────────┘
               │           └── email_sender.py ──────────┘
               │
               ├── face_detection.py (get_face_app — for enrollment)
               └── utils.py (load_known_faces)
```

### Who owns what

| Responsibility | Owner |
|---------------|-------|
| All detection logic (face + weapon + person + alarm + email) | `pipeline.py` |
| Face detection + recognition | `face_detection.py` |
| Weapon detection | `weapon_detection.py` |
| Person detection | `person_detection.py` |
| Alarm trigger/stop | `alarm.py` |
| Email sending | `email_sender.py` |
| Embedding generation (enrollment) | `app.py` → `get_face_app()` |
| Embedding loading + validation | `utils.py` |
| All configuration constants | `config.py` |
| Web routes + MJPEG stream | `app.py` |
| CLI camera loop | `main.py` → `pipeline.py` |

### Key design decisions

**`pipeline.py` is the single source of truth.** Before this file existed, the entire ~120-line detection loop was copy-pasted between the CLI and Flask server. Any bug fix had to be applied twice. Now there is exactly one place where detection, alarm, and email logic lives. Both `app.py` and `main.py` delegate to it completely.

**`face_detection.py` exposes `get_face_app()` publicly.** The same `FaceAnalysis` singleton is used for both live detection (pipeline) and enrollment (app.py). This guarantees enrollment and inference use identical models and produce identically-dimensioned vectors. Previously they used different models, which silently broke recognition entirely.

---

## 5. Module Reference

### 5.1 `config.py`

Central configuration hub. Every tuneable constant in the entire system is defined here and backed by an environment variable so nothing needs to be changed in source code between deployments.

**How it works:**
```python
import os
from dotenv import load_dotenv
load_dotenv()                          # reads .env file if present
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "2"))   # env var or default
```

**All constants:**

| Constant | Env Variable | Type | Default | Description |
|----------|-------------|------|---------|-------------|
| `CAMERA_INDEX` | `CAMERA_INDEX` | int | `2` | OpenCV camera device index. `0` = first camera, `1` = second, etc. Change if the wrong camera opens. |
| `FACE_THRESHOLD` | `FACE_THRESHOLD` | float | `0.35` | Minimum cosine similarity score (0.0–1.0) to accept a face as "known". Lower = more permissive, higher = stricter. ArcFace same-person scores typically 0.3–0.7. |
| `WEAPON_THRESHOLD` | `WEAPON_THRESHOLD` | float | `0.60` | Minimum YOLO confidence (0.0–1.0) to report a weapon detection. |
| `PERSON_THRESHOLD` | `PERSON_THRESHOLD` | float | `0.50` | Minimum YOLO confidence (0.0–1.0) to report a person detection. |
| `ALARM_COOLDOWN` | `ALARM_COOLDOWN` | int | `5` | Seconds to wait before re-triggering the alarm in the same threat event. Prevents the alarm from firing hundreds of times per second. |
| `ALARM_EMAIL_THRESHOLD` | `ALARM_EMAIL_THRESHOLD` | int | `3` | How many times the alarm must fire before an email is sent. Prevents a momentary false detection from immediately sending email. |
| `FRAME_SKIP` | `FRAME_SKIP` | int | `3` | Process every Nth frame. `1` = every frame (maximum accuracy, heaviest CPU). `3` = every 3rd frame (3× lighter, recommended for CPU). |
| `EMBEDDINGS_FILE` | `EMBEDDINGS_FILE` | str | `facenet_embeddings1.npy` | Path to the numpy file storing enrolled face embeddings. |
| `DATASET_FOLDER` | `DATASET_FOLDER` | str | `dataset` | Root directory for enrolled person image folders. |
| `EMAIL_SENDER` | `ALERT_SENDER_EMAIL` | str | None | Gmail address that sends the alerts. |
| `EMAIL_PASSWORD` | `ALERT_EMAIL_PASSWORD` | str | None | Gmail App Password (16 characters). NOT your regular Gmail password. |
| `EMAIL_RECIPIENT` | `ALERT_TO_EMAIL` | str | None | Email address that receives alerts. |

`config.py` also calls `logging.basicConfig(...)` which sets up formatted logging for the entire application in one place.

---

### 5.2 `pipeline.py`

The heart of the system. Contains all detection coordination, alarm logic, email triggering, and both camera loops. Everything funnels through this file.

#### `DetectionState` dataclass

```python
@dataclass
class DetectionState:
    alarm_playing:   bool  = False   # Is the alarm currently sounding?
    last_alarm_time: float = 0.0     # Unix timestamp of the last alarm trigger
    alarm_count:     int   = 0       # How many times the alarm has fired this session
```

One instance is created per camera session and passed into every `process_frame()` call. This keeps alarm state persistent across frames without using global variables.

#### `process_frame(frame, known_faces, state) → (bool, bool, bool)`

The core function. Called on every processed frame. Does the following in order:

1. Spawns a background thread for face detection
2. Spawns a background thread for weapon detection
3. Runs person detection on the calling thread (overlapping with step 1 & 2)
4. Joins both background threads (waits for them to complete)
5. Draws person bounding boxes in yellow
6. Evaluates the three-way threat condition
7. Triggers alarm and/or email if threat is detected
8. Stops alarm if threat has cleared
9. Returns `(person_detected, unknown_detected, weapon_detected)`

#### `_open_camera() → cv2.VideoCapture`

Opens the camera at `CAMERA_INDEX`, sets resolution to 640×480, and logs a helpful error message if the camera cannot be opened (instead of crashing silently).

#### `detect_objects_in_realtime()`

The CLI camera loop:
- Opens camera with `_open_camera()`
- Loads `known_faces` from disk
- Loops: read frame → skip if not Nth frame → reload known_faces every 30 s → `process_frame()` → `cv2.imshow()` → check for 'q' keypress
- Cleans up on exit: releases camera, destroys windows, stops alarm

#### `generate_frames(stop_event, streaming_flag)`

The Flask MJPEG generator:
- Opens camera with `_open_camera()`
- Loads `known_faces` from disk
- Loops: read frame → skip if not Nth frame → reload known_faces every 30 s → `process_frame()` → `cv2.imencode()` → `yield` MJPEG chunk
- Stops when `stop_event` is set or `streaming_flag()` returns False
- Cleans up: releases camera, stops alarm

**Why the 30-second reload?** `known_faces` is loaded once at stream start. If a user enrolls a new face while the stream is running, the detection would never see the new person without a reload. The 30-second periodic reload picks up changes automatically.

---

### 5.3 `face_detection.py`

Handles all face-related work: finding faces in a frame and deciding if each face belongs to an enrolled person.

#### The InsightFace singleton

```python
_face_app = None

def get_face_app():
    global _face_app
    if _face_app is not None:
        return _face_app
    from insightface.app import FaceAnalysis
    _face_app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
    _face_app.prepare(ctx_id=0, det_size=(320, 320))
    return _face_app
```

`get_face_app()` is exposed publicly so `app.py`'s enrollment code can call it directly. This ensures both paths (detection and enrollment) share the exact same model instance, guaranteeing that embeddings generated during enrollment are in the same vector space as embeddings generated during live detection.

`det_size=(320, 320)` is deliberately smaller than the default `(640, 640)`. At 640×480 camera resolution, a 320×320 detection grid still catches faces down to approximately 20×20 pixels, which is sufficient for surveillance-range distances. The smaller grid runs roughly 2× faster on CPU.

#### `_cosine_sim(a, b) → float`

Pure numpy cosine similarity implementation:
```
cosine_sim(a, b) = dot(a, b) / (||a|| × ||b||)
```

This works correctly regardless of whether the vectors are L2-normalized. The InsightFace buffalo_s `w600k_mbf.onnx` model on some ONNX runtime versions outputs raw (un-normalized) vectors with norm ≈ 20–30. The explicit division by both norms makes the comparison scale-invariant.

Returns `0.0` if either vector is zero to prevent division by zero.

#### `detect_faces(frame, known_faces, threshold) → bool`

Step-by-step:

1. `app.get(frame)` — runs the SCRFD-500M detector to find all faces, then runs ArcFace to embed each one. Returns a list of `Face` objects.
2. If no faces found → return False immediately
3. For each face:
   - Extract `face.embedding` (512-d vector)
   - Extract `face.bbox` (x1, y1, x2, y2 bounding box coordinates)
   - Loop over `known_faces` dict, computing cosine similarity with each stored embedding
   - Track the best match (highest cosine similarity score)
   - If `best_score < threshold` → label as "Unknown", mark `unknown_detected = True`, draw red box
   - If `best_score >= threshold` → label as person's name with score, draw green box
4. Return `unknown_detected`

The label drawn on the frame is `"PersonName (0.52)"` showing the match confidence, which helps tune the threshold.

---

### 5.4 `weapon_detection.py`

Runs the custom-trained YOLO weapon detector on each frame.

#### Model loading

```python
yolo_model = YOLO("best100.pt")
classes = _load_classes("coco2.txt")
```

Both are loaded **once at module import time**, not per frame. This means the ~100–200ms model-loading overhead is paid only when Python first imports the module (at app startup), not on every detection call.

#### `detect_weapons(frame) → bool`

Step-by-step:

1. `yolo_model(frame, verbose=False)` — runs inference. `verbose=False` suppresses YOLO's per-frame console output.
2. For each detection box:
   - Check confidence: skip if below `WEAPON_THRESHOLD` (0.60)
   - Look up class name from `coco2.txt` using `cls_idx`
   - Check if class name contains `"gun"` or `"knife"` (substring match, case-insensitive — both are lower-cased at load time)
   - If both checks pass: draw a red bounding box with `"classname conf"` label, set `weapon_detected = True`
3. Return `weapon_detected`

The two-stage filter (confidence AND class name check) means only high-confidence weapon detections are reported. Other YOLO detections (if the model has non-weapon classes) are silently ignored.

---

### 5.5 `person_detection.py`

Runs YOLOv8n to detect people in the frame.

#### `detect_people(frame) → list[tuple[int,int,int,int]]`

Step-by-step:

1. `model(frame, verbose=False)[0]` — runs YOLOv8n inference, takes the first (and only) result object
2. For each detected box:
   - Check class: must be `0` (COCO class 0 is "person")
   - Check confidence: must be ≥ `PERSON_THRESHOLD` (0.50)
   - If both pass: append `(x1, y1, x2, y2)` to the list
3. Return the list of person bounding boxes

**Critically, this module does NOT draw on the frame.** Drawing is handled by `pipeline.py` in yellow so the colour is consistent regardless of internal module behaviour. This is the opposite design from `face_detection.py` and `weapon_detection.py` which draw their own boxes — person boxes are drawn after the parallel threads rejoin.

---

### 5.6 `app.py`

The Flask web server. Handles all HTTP routes and the face enrollment workflow.

#### Application setup

```python
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB upload cap
os.makedirs(DATASET_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
```

#### Stream state management

Two module-level variables control the stream:
- `streaming: bool` — False signals the generator to stop
- `stop_event: threading.Event` — set by `/stop_stream`, checked by `generate_frames()` on every iteration

#### `_allowed_file(filename) → bool`

Checks that the uploaded file has an allowed extension. The check uses `rsplit(".", 1)[1].lower()` to handle filenames with multiple dots correctly (e.g. `photo.backup.jpg` → extension is `jpg`).

#### `_update_embeddings()`

The face enrollment engine. Called in a background thread after images are uploaded:

1. Load existing `facenet_embeddings1.npy` if it exists (preserving people not in the current upload batch)
2. For every subfolder in `dataset/`:
   - Read each image with `cv2.imread()`
   - Run `face_app.get(img)` to detect faces and generate ArcFace embeddings
   - Collect `faces[0].embedding` for each image where a face was found
   - Skip images where no face is detected (logs a warning)
3. Average all embeddings for each person: `np.mean(embeddings, axis=0)`
   - Averaging multiple embeddings from different photos of the same person creates a more robust "centroid" representation
4. Save the updated dict to `facenet_embeddings1.npy` under `_embed_lock`

The `_embed_lock` protects the `.npy` file from simultaneous read/write if the stream happens to reload at the same moment enrollment is saving.

#### `_run_update_embeddings_bg()`

A thin wrapper around `_update_embeddings()` that sets/clears the `_embed_running` flag. If the user clicks "Upload" twice quickly, the second request checks `_embed_running` and returns a message saying an update is already in progress rather than launching two concurrent background threads.

#### Routes

| Route | Method | What it does |
|-------|--------|-------------|
| `/` | GET | Renders `index.html` (home page with three navigation cards) |
| `/detection` | GET | Renders `detection.html` (Start/Stop Streaming buttons, `<img>` tag for MJPEG) |
| `/video_feed` | GET | Clears `stop_event`, sets `streaming=True`, returns a `multipart/x-mixed-replace` streaming response from `generate_frames()` |
| `/stop_stream` | GET | Sets `streaming=False`, calls `stop_event.set()`, returns plain text "Streaming stopped" |
| `/safepeople` | GET | Scans `dataset/` for person subfolders, renders `safepeople.html` with list |
| `/create_encoding` | GET | Renders the add/delete form |
| `/create_encoding` | POST `action=add` | Saves uploaded images → launches background embedding update |
| `/create_encoding` | POST `action=delete` | Removes person folder + entry from `.npy` file |

#### Security measures in file upload

```python
safe_person  = secure_filename(person_name)        # sanitises folder name
person_folder = os.path.join(DATASET_FOLDER, safe_person)

for img in images:
    if img and _allowed_file(img.filename):
        img.save(os.path.join(person_folder, secure_filename(img.filename)))
```

`secure_filename()` from Werkzeug strips path separators, null bytes, and other dangerous characters. An attacker who submits `person_name = "../../etc"` gets a sanitised name like `etc`. Both the folder name and every individual filename are sanitised independently.

---

### 5.7 `main.py`

```python
from pipeline import detect_objects_in_realtime

if __name__ == "__main__":
    detect_objects_in_realtime()
```

Intentionally minimal. All detection logic lives in `pipeline.py`. The CLI entry point is five lines so there is nothing to maintain here.

---

### 5.8 `utils.py`

#### `load_known_faces(embeddings_file) → dict`

Loads the enrolled face embeddings with full defensive error handling. Returns an empty dict (never raises) in all failure cases so the rest of the system keeps running even if the embeddings file is broken.

Failure cases handled:
- File does not exist → log warning, return `{}`
- File exists but `np.load()` fails → log error, return `{}`
- Loaded object is not a dict → log error, return `{}`
- Any embedding has wrong dimension (not 512-d) → log warning, return `{}`

The dimension check catches embeddings from the old `keras-facenet` model which produced 128-d vectors. These are incompatible with ArcFace 512-d vectors.

**Why no L2-norm check?** Earlier versions of this code checked `if norm > 2.0: return {}` to catch old Facenet512 embeddings (which have norm ≈ 15–30). This was removed because InsightFace buffalo_s `w600k_mbf.onnx` also outputs un-normalised vectors with norm ≈ 20–30 on some ONNX runtime versions. The norm check was incorrectly rejecting valid ArcFace embeddings, causing every face to show as Unknown. The correct fix is: since enrollment now exclusively uses InsightFace, any 512-d embedding stored by the current code is valid. Cosine similarity handles arbitrary-norm vectors correctly via explicit normalisation.

---

### 5.9 `alarm.py`

#### Lazy initialisation pattern

```python
_mixer_ready = False
_alarm_sound = None

def _init_mixer() -> bool:
    global _mixer_ready, _alarm_sound
    if _mixer_ready:
        return True
    try:
        import pygame
        pygame.mixer.init()
        _alarm_sound = pygame.mixer.Sound("alarm.wav")
        _mixer_ready = True
    except Exception as exc:
        logger.warning(f"Audio not available — alarm will be silent. Reason: {exc}")
        _mixer_ready = False
    return _mixer_ready
```

`pygame` is imported inside `_init_mixer()`, not at the top of the file. This means `import alarm` never calls `pygame.mixer.init()`. If the server has no audio device (headless Linux server, Docker container), the `except` block catches the error and logs a warning. The rest of the application continues running — it just won't make any sound. This is graceful degradation.

`_init_mixer()` is called on every `start_alarm()` call but immediately returns `True` after the first successful initialisation (the `if _mixer_ready: return True` guard at the top).

#### `start_alarm()`
Calls `_init_mixer()`, then `_alarm_sound.play(maxtime=5000)`. The `maxtime=5000` means the sound plays for at most 5 seconds even if `stop_alarm()` is never called.

#### `stop_alarm()`
Checks `_mixer_ready` before calling `.stop()` to avoid errors if the mixer was never initialised.

---

### 5.10 `email_sender.py`

#### `send_email_with_attachment(image_path, to_email=None) → bool`

Builds and sends a MIME email over Gmail SMTP SSL. Returns `True` on success, `False` on any failure. Never raises — the caller (`pipeline.py`) does not need to handle exceptions.

**Failure modes handled:**
- Any of `EMAIL_SENDER`, `EMAIL_PASSWORD`, `EMAIL_RECIPIENT` is None → log warning, return False (happens when `.env` is not configured)
- `image_path` is None or file does not exist → email sent without attachment (logs a warning)
- SMTP connection fails, login fails, send fails → log error, return False

**Email construction:**
```
From:     EMAIL_SENDER
To:       to_email argument or EMAIL_RECIPIENT
Subject:  "Security Alert: Unknown Person with Weapon Detected"
Body:     Plain text description
Attach:   screenshot.png (if path exists)
```

**Gmail SMTP settings:**
- Server: `smtp.gmail.com`
- Port: `465` (SMTP_SSL — encrypts the entire connection from the start)
- Auth: Gmail App Password (16 characters) — required because Google disabled plain-password login for programmatic access

---

## 6. Detection Pipeline — Deep Dive

### Frame lifecycle

```
cap.read()                   # grab raw BGR frame from camera driver
    │
    ├─ frame_count += 1
    │
    ├─ if frame_count % FRAME_SKIP != 0: continue   # skip this frame
    │
    ├─ if time.time() - last_reload >= 30:
    │       known_faces = load_known_faces(EMBEDDINGS_FILE)   # periodic refresh
    │
    └─ process_frame(frame, known_faces, state)
           │
           ├─ Thread: _run_face()
           │     detect_faces(frame, known_faces, FACE_THRESHOLD)
           │         → app.get(frame)   [SCRFD + ArcFace]
           │         → cosine_sim vs known_faces
           │         → draw boxes on frame
           │         → result["face"] = unknown_detected
           │
           ├─ Thread: _run_weapon()
           │     detect_weapons(frame)
           │         → yolo_model(frame)
           │         → filter by conf and class
           │         → draw boxes on frame
           │         → result["weapon"] = weapon_detected
           │
           ├─ Calling thread: detect_people(frame)
           │     model(frame)
           │     → filter class=0, conf>=0.5
           │     → return person_boxes
           │
           ├─ face_t.join(); weapon_t.join()
           │
           ├─ Draw person boxes (yellow) on frame
           │
           ├─ Evaluate threat = person AND unknown AND weapon
           │
           └─ Handle alarm + email (see Section 10)
```

### FRAME_SKIP explained

The camera produces frames at its native framerate (typically 25–30 fps). With `FRAME_SKIP=3`, only every 3rd frame is processed:

```
Frame 1: skip
Frame 2: skip
Frame 3: PROCESS ← face + weapon + person detection runs here
Frame 4: skip
Frame 5: skip
Frame 6: PROCESS
...
```

This reduces CPU usage approximately 3× with minimal impact on real-world threat detection, since threats typically persist for multiple seconds. The skipped frames are NOT displayed — the stream still runs at camera framerate, only detection is throttled.

---

## 7. Face Recognition Pipeline — Deep Dive

### The InsightFace buffalo_s pack

When InsightFace initialises with `name="buffalo_s"`, it downloads (on first run) and loads five ONNX models:

| File | Architecture | Purpose | Input size |
|------|-------------|---------|-----------|
| `det_500m.onnx` | SCRFD-500M | Face detection — finds bounding boxes | Dynamic (set by det_size) |
| `w600k_mbf.onnx` | ArcFace + MobileNetV1 | Face recognition — generates 512-d embedding | 112×112 |
| `1k3d68.onnx` | 3DDFA-v2 variant | 3D 68-point landmark detection | 192×192 |
| `2d106det.onnx` | 2D landmark model | 2D 106-point landmark detection | 192×192 |
| `genderage.onnx` | ShuffleNet | Gender + age estimation | 96×96 |

The landmarks are used internally by InsightFace to align the face crop before passing it to the ArcFace model. Proper alignment (rotating the face so eyes are horizontal) significantly improves recognition accuracy.

### SCRFD — Sample and Computation Redistribution for Efficient Face Detection

SCRFD-500M is a 500 MFLOPs face detector designed specifically for real-time use. Compared to the old Haar cascade:

| Property | Haar Cascade | SCRFD-500M |
|----------|-------------|-----------|
| Method | Hand-crafted features | Deep CNN |
| Rotation robustness | Frontal only (~±30°) | Up to ±90° yaw |
| Occlusion handling | Poor | Good |
| Small faces | Poor below 40px | Good down to ~20px |
| False positive rate | High | Low |
| Speed | Very fast | Fast (CNN, but optimised) |

### ArcFace — Additive Angular Margin Loss

ArcFace trains a CNN (here: MobileNetV1) to produce face embeddings such that:
- Same person's embeddings have high cosine similarity (small angle in embedding space)
- Different people's embeddings have low cosine similarity (large angle)

The "angular margin" in the loss function directly optimises the angular separation between classes, making it more discriminative than older metric learning approaches.

**w600k_mbf**: Trained on WebFace600K dataset with MobileNetV1 backbone. The `mbf` suffix stands for MobileNet-based Face. It is the lightweight model in the buffalo_s pack, optimised for CPU inference.

### Cosine similarity as a matching metric

```
score = dot(query_embedding, stored_embedding) / (||query|| × ||stored||)
```

Score ranges from -1.0 (completely opposite) to +1.0 (identical direction). In practice:
- Same person, similar conditions: 0.4–0.8
- Same person, different lighting/angle: 0.2–0.5
- Different people: -0.1–0.2 (rarely above 0.3)

The default threshold of **0.35** sits in the gap between same-person and different-person distributions. If you get too many false unknowns, lower it to 0.25–0.30. If you get too many false recognitions (wrong person accepted), raise it to 0.40–0.45.

### Embedding averaging during enrollment

When multiple photos are uploaded for one person:
```python
embeddings = [face_app.get(img)[0].embedding for img in images if face detected]
stored_embedding = np.mean(embeddings, axis=0)
```

The mean of multiple face embeddings from the same person creates a "centroid" in embedding space that is more representative than any single photo. It smooths out pose, lighting, and expression variation. More enrollment photos → more robust centroid → better live recognition.

**Recommended enrollment:** 3–8 photos per person, with variation in:
- Lighting (bright, dim, natural, artificial)
- Angle (slight left, right, straight-on)
- Expression (neutral, slight smile)

---

## 8. Weapon Detection Pipeline — Deep Dive

### The custom model: best100.pt

`best100.pt` is a YOLO11s model (small variant of YOLO11) trained specifically on a weapon dataset for 100 epochs. It was trained using Ultralytics with a custom dataset containing gun and knife images.

**Why a custom model instead of COCO?**
- Standard COCO-pretrained YOLO models include "knife" as a class but do not include "gun" as a distinct weapon class (COCO has no gun class)
- The custom training dataset focused specifically on the types of weapons relevant to security surveillance
- 100 epochs gives strong convergence for a specialised detection task

### Detection filter logic

```python
for idx in range(len(result.boxes)):
    conf = float(result.boxes.conf[idx])
    if conf < WEAPON_THRESHOLD:           # Step 1: confidence gate
        continue

    cls_idx    = int(result.boxes.cls[idx])
    class_name = classes[cls_idx]         # Step 2: look up class name

    if "gun" not in class_name and "knife" not in class_name:  # Step 3: class gate
        continue

    weapon_detected = True
    # draw red box...
```

The substring check (`"gun" in class_name`) handles class names like `"handgun"`, `"rifle"`, `"assault_gun"`, etc. without needing to enumerate every variant.

### Training artifacts

The training run that produced `best100.pt` generated metrics stored in `runs100epochs/` (now removed from the repo — the trained model is all that is needed for deployment). Key training results:
- Trained for 100 epochs
- Training metrics (F1, precision-recall curves) confirmed good convergence

---

## 9. Person Detection Pipeline — Deep Dive

### YOLOv8n on COCO

`yolov8n.pt` is the nano (smallest) variant of YOLOv8, pretrained on the COCO dataset. COCO class 0 is "person". The model was trained on 118,000 images with 250,000 person instances.

YOLOv8n was chosen over the previously-used YOLOv5su because:

| Property | YOLOv5su | YOLOv8n |
|----------|---------|---------|
| Parameters | ~7.2M | ~3.2M |
| Inference speed (CPU) | ~80ms | ~40–50ms |
| mAP50 (person class) | Comparable | Comparable |
| File size | ~14MB | ~6MB |

YOLOv8n gives similar person detection accuracy at roughly half the CPU cost and half the disk space.

### Why person detection?

The three-way threat condition requires a person to be present. This prevents the system from alarming on:
- An unattended weapon left on a surface (no person present)
- A known-face person holding a weapon (person present but face is known)

The person detection adds a spatial sanity check. In practice, if there's a face detected there's almost certainly a person, but the explicit person check ensures the bounding box geometry is correct.

---

## 10. Alarm & Alert System

### Alarm trigger sequence

```
process_frame() evaluates threat = True
       │
       ├─ Is alarm already playing? No → proceed
       │  OR has ALARM_COOLDOWN seconds passed since last alarm? → proceed
       │
       ├─ threading.Thread(target=start_alarm).start()
       │       │
       │       └─ alarm.py: _init_mixer() → pygame.mixer.Sound.play(maxtime=5000)
       │
       ├─ state.alarm_playing = True
       ├─ state.last_alarm_time = now
       └─ state.alarm_count += 1
              │
              ├─ state.alarm_count < ALARM_EMAIL_THRESHOLD → no email yet
              │
              └─ state.alarm_count >= ALARM_EMAIL_THRESHOLD
                     │
                     ├─ snapshot = frame.copy()   ← copy frame BEFORE yielding thread
                     ├─ threading.Thread(target=_send, args=(snapshot,)).start()
                     │       │
                     │       ├─ cv2.imwrite("screenshot.png", img)
                     │       └─ send_email_with_attachment("screenshot.png")
                     │
                     └─ state.alarm_count = 0   ← reset prevents email every frame
```

### Alarm cooldown

The `ALARM_COOLDOWN` (default 5 seconds) prevents the alarm from re-triggering hundreds of times per second. On a 30 fps camera with FRAME_SKIP=3, there are ~10 processed frames per second. Without the cooldown, the alarm would be started and stopped 10 times per second, which would produce a stuttering effect and cause `alarm_count` to reach `ALARM_EMAIL_THRESHOLD` almost instantly.

### Email threshold

`ALARM_EMAIL_THRESHOLD` (default 3) means the alarm must fire 3 times before an email is sent. With `ALARM_COOLDOWN=5`, that means the threat must persist for at least ~10–15 seconds before an email fires. This prevents a momentary glitch from sending email but still responds quickly to sustained threats.

After sending, `alarm_count` resets to 0. If the threat persists, the count climbs again and another email fires after another 3 alarm cycles (~10–15 more seconds). This provides ongoing notification without flooding the recipient's inbox.

### Threat clearance

```python
elif state.alarm_playing:
    stop_alarm()
    state.alarm_playing = False
```

As soon as `threat = False` (any of the three conditions drops), the alarm stops. `alarm_count` is NOT reset when the threat clears — it only resets after an email fires. This means a series of brief threat events accumulate toward the email threshold.

---

## 11. Web Interface

### Home page (`/`)

Three navigation cards linking to the three main sections: Start Detection, Safe People, Generate Encodings. Uses custom CSS (`styles.css`) with a glowing animated dot background effect.

### Detection page (`/detection`)

Contains a hidden `<img>` element that points to `/video_feed`. Clicking "Start Streaming" sets `img.src = '/video_feed'`, which causes the browser to open an HTTP connection to Flask. Flask responds with a `multipart/x-mixed-replace` stream — the browser continuously reads JPEG frames from this stream and displays them as a live video.

Clicking "Stop Streaming" clears `img.src` (stopping the browser's HTTP request) and calls `/stop_stream` (which signals the server-side generator to stop and release the camera).

### MJPEG streaming protocol

```
HTTP Response Headers:
  Content-Type: multipart/x-mixed-replace; boundary=frame

Response Body (continuous stream):
  --frame\r\n
  Content-Type: image/jpeg\r\n
  \r\n
  [JPEG binary data]\r\n
  --frame\r\n
  Content-Type: image/jpeg\r\n
  \r\n
  [JPEG binary data]\r\n
  ...
```

Every time `generate_frames()` yields a chunk, Flask writes it to the HTTP response. The browser's `<img>` tag natively understands this format and updates the displayed image with each new JPEG frame, creating the appearance of a live video stream.

### Enrollment page (`/create_encoding`)

Two-panel form:
- **Add panel**: text input for name + file picker for images + submit. Shows an animated progress bar while the form submits (purely cosmetic — the embedding runs in the background after the HTTP response returns).
- **Delete panel**: text input for name + delete button

Messages are colour-coded: green for success (✅), red for errors (❌). This is done with Jinja2 conditional class:
```html
{{ 'text-red-400' if message.startswith('❌') else 'text-green-400' }}
```

### Safe people page (`/safepeople`)

Scans the `dataset/` folder and lists every person who has a subfolder with images. Displayed as a table.

---

## 12. Data Flow Diagrams

### Enrollment flow

```
User at /create_encoding
    │
    │ POST action=add, person_name="Alice", images=[photo1.jpg, photo2.jpg, photo3.jpg]
    ▼
app.py: /create_encoding handler
    │
    ├─ secure_filename("Alice") → "Alice"
    ├─ os.makedirs("dataset/Alice", exist_ok=True)
    ├─ Save photo1.jpg → dataset/Alice/photo1.jpg
    ├─ Save photo2.jpg → dataset/Alice/photo2.jpg
    ├─ Save photo3.jpg → dataset/Alice/photo3.jpg
    │
    ├─ if not _embed_running:
    │       threading.Thread(target=_run_update_embeddings_bg).start()
    │               │
    │               └─ _update_embeddings()
    │                       │
    │                       ├─ database = np.load("facenet_embeddings1.npy")  [existing]
    │                       │
    │                       ├─ For person "Alice" in dataset/:
    │                       │     photo1.jpg → face_app.get(img) → embedding_1 (512-d)
    │                       │     photo2.jpg → face_app.get(img) → embedding_2 (512-d)
    │                       │     photo3.jpg → face_app.get(img) → embedding_3 (512-d)
    │                       │     database["Alice"] = mean([e1, e2, e3])
    │                       │
    │                       └─ np.save("facenet_embeddings1.npy", database)
    │
    └─ Return HTTP 200 with success message
           "Alice added (3 images). Embedding generation started in background."
```

### Live detection flow (web)

```
Browser: GET /video_feed
    │
    ▼
app.py: video_feed()
    │
    └─ Response(generate_frames(stop_event, lambda: streaming), mimetype="multipart/...")
              │
              ├─ cap = cv2.VideoCapture(CAMERA_INDEX)
              ├─ known_faces = load_known_faces("facenet_embeddings1.npy")
              ├─ state = DetectionState()
              │
              └─ LOOP:
                    ret, frame = cap.read()
                    │
                    ├─ Every 30s: known_faces = load_known_faces(...)  [refresh]
                    │
                    ├─ process_frame(frame, known_faces, state)
                    │        │
                    │        ├─ Thread: detect_faces() → draws green/red boxes
                    │        ├─ Thread: detect_weapons() → draws red boxes
                    │        └─ Caller: detect_people() → returns boxes
                    │                  pipeline draws yellow boxes
                    │
                    │        ├─ threat = person AND unknown AND weapon?
                    │        ├─ If yes: alarm + (maybe email)
                    │        └─ If alarm was playing but threat gone: stop alarm
                    │
                    ├─ cv2.imencode(".jpg", frame) → JPEG bytes
                    │
                    └─ yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + bytes + b"\r\n"
                                │
                                └─ → Browser receives JPEG → renders in <img> tag
```

---

## 13. Threading Model

The system uses Python threads in two places:

### 1. Per-frame detector parallelism (inside `process_frame`)

```
Calling thread ──────────────────────────────────────────────────────────►
               spawn face_t    join face_t
                    │               ▲
                    └── face_t ─────┘  (InsightFace SCRFD + ArcFace)
               spawn weapon_t  join weapon_t
                    │               ▲
                    └── weapon_t ───┘  (YOLO weapon detection)
               detect_people() ──►
               (runs concurrently with above two)
```

Wall-clock time ≈ `max(face_time, weapon_time) + person_time`  
Instead of: `face_time + weapon_time + person_time`

Since face detection (~150ms) and weapon detection (~100ms) overlap, the effective per-frame cost is ~150ms + 50ms = ~200ms rather than ~300ms.

### 2. Background tasks

| Task | When spawned | Why background? |
|------|-------------|----------------|
| `start_alarm()` | Every alarm trigger | pygame audio must not block the detection loop |
| `_send(img)` | Every email trigger | SMTP connect+auth takes 1–5 seconds |
| `_run_update_embeddings_bg()` | After image upload | Embedding generation takes 10–60 seconds depending on dataset size |

### GIL and OpenCV

Python's GIL (Global Interpreter Lock) prevents true CPU parallelism for pure Python code. However, both InsightFace (via ONNX Runtime) and YOLO (via PyTorch) release the GIL during C++ inference. This means face detection and weapon detection genuinely run in parallel at the CPU level during their inference phases.

The `threading.Lock()` in `process_frame()` protects the `result` dict, which is Python-level shared state. While GIL-protected dict writes are technically atomic, the explicit lock is correct practice since the threads also call into C extension code that releases the GIL.

---

## 14. Configuration Reference

All values can be set in `.env` or as system environment variables. Values in `.env` take precedence over system environment variables for keys defined in `.env`; system env vars take precedence over `.env` defaults.

### Setting values

**Option A — `.env` file (recommended):**
```
CAMERA_INDEX=0
FACE_THRESHOLD=0.30
FRAME_SKIP=5
ALERT_SENDER_EMAIL=myemail@gmail.com
ALERT_EMAIL_PASSWORD=abcd efgh ijkl mnop
ALERT_TO_EMAIL=security@example.com
```

**Option B — PowerShell environment variables (session-scoped):**
```powershell
$env:CAMERA_INDEX = "0"
$env:FACE_THRESHOLD = "0.30"
python app.py
```

### Tuning guide

**Camera is wrong device:**  
Try `CAMERA_INDEX=0`, `1`, `2` until the correct camera opens.

**Too many false "Unknown" faces (known people not recognised):**  
Lower `FACE_THRESHOLD` to `0.28`–`0.30`. Also upload more enrollment photos.

**Unknown people being accepted as known (wrong person recognised):**  
Raise `FACE_THRESHOLD` to `0.42`–`0.45`.

**System too slow / high CPU:**  
Raise `FRAME_SKIP` to `5` or `6`. This processes fewer frames per second but reduces CPU load proportionally.

**Alarm fires too easily on brief glitches:**  
Raise `ALARM_EMAIL_THRESHOLD` to `5`. Raise `ALARM_COOLDOWN` to `8`–`10`.

**Weapons not being detected:**  
Lower `WEAPON_THRESHOLD` to `0.45`–`0.50`.

**Too many false weapon detections:**  
Raise `WEAPON_THRESHOLD` to `0.70`–`0.75`.

---

## 15. Models Reference

### InsightFace buffalo_s (face pipeline)

- **Downloaded to:** `~/.insightface/models/buffalo_s/` on first run (~85 MB total)
- **Detection model:** SCRFD-500M (`det_500m.onnx`)
  - Architecture: SCRFD (Sample and Computation Redistribution)
  - Input: variable size (set to 320×320 for speed)
  - Output: face bounding boxes + landmark points + detection confidence
- **Recognition model:** ArcFace MobileNetV1 (`w600k_mbf.onnx`)
  - Training dataset: WebFace600K (600K identity images)
  - Backbone: MobileNetV1
  - Output: 512-dimensional face embedding vector
  - Note: outputs are NOT L2-normalised on all ONNX runtime versions (norm ≈ 20–30 is normal and correct — cosine similarity handles this)

### best100.pt (weapon detection)

- **Architecture:** YOLO11s (small variant of YOLO11)
- **Training:** 100 epochs on a custom weapon dataset
- **Classes:** Defined in `coco2.txt` (gun, knife, and potentially other classes)
- **Active classes:** Only `"gun"` and `"knife"` substrings trigger the weapon flag

### yolov8n.pt (person detection)

- **Architecture:** YOLOv8n (nano — smallest YOLOv8 variant)
- **Training:** COCO dataset (pretrained, not fine-tuned)
- **Active class:** Class 0 only ("person")
- **File:** Auto-downloaded by Ultralytics if not present

---

## 16. Environment Setup & Installation

### Requirements

- Windows 10/11 (also works on Linux/macOS with minor path adjustments)
- Python 3.11+ (tested on 3.13)
- Webcam or USB camera
- Internet connection (first run downloads InsightFace buffalo_s models ~85 MB)

### Step-by-step installation

```powershell
# 1. Navigate to the project directory
cd "d:\coding\.vscode\project\Real-Time-Weapon-Face-Detection"

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
.\.venv\Scripts\Activate.ps1

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install PyTorch (CPU build — works on all machines)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

#    For NVIDIA GPU (replace cu121 with your CUDA version):
#    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 6. Install all remaining dependencies
pip install -r requirements.txt

# 7. Configure credentials
copy .env.example .env
notepad .env
#    Fill in ALERT_SENDER_EMAIL, ALERT_EMAIL_PASSWORD, ALERT_TO_EMAIL

# 8. Verify everything loads
python -c "import app; print('All imports OK')"
```

### Verify environment variables loaded

```powershell
python scripts\check_env.py
```

Expected output:
```
dotenv.load_dotenv() -> True
.env exists = True
ALERT_SENDER_EMAIL=  youremail@gmail.com
ALERT_EMAIL_PASSWORD= ***
ALERT_TO_EMAIL=  recipient@gmail.com
```

### Test email sending

```powershell
python scripts\send_test_email.py
```

---

## 17. Running the System

### Web UI mode (recommended)

```powershell
python app.py
```

Output:
```
12:00:00 [INFO] weapon_detection: ...YOLO loading...
12:00:02 [INFO] person_detection: ...YOLO loading...
12:00:02 [INFO] utils: Loaded 1 enrolled face(s): ['Alice']
 * Running on http://127.0.0.1:5000
```

Open `http://127.0.0.1:5000` in your browser. Click **Start Detection** → **Start Streaming**.

### CLI mode

```powershell
python main.py
```

An OpenCV window named "SafeGuard — Real-time Detection (q = quit)" opens showing the live annotated feed. Press `q` to exit cleanly.

### Stopping cleanly

- **Web UI:** Click "Stop Streaming" in the browser, then Ctrl+C in the terminal
- **CLI:** Press `q` in the OpenCV window

If the app crashes or is killed, the camera may remain locked. Run:
```powershell
taskkill /F /IM python.exe
```
before restarting.

---

## 18. Enrolling Safe People

### How to add a person

1. Start the app: `python app.py`
2. Open `http://127.0.0.1:5000/create_encoding`
3. Enter the person's name (e.g. `Alice`)
4. Upload 3–8 clear face photos (JPG or PNG, max 10 MB each)
5. Click **Upload & Update Embeddings**
6. The server responds immediately. Background embedding generation runs for a few seconds
7. The face is now enrolled — the stream auto-reloads within 30 seconds

### Photo guidelines for best recognition accuracy

| Guideline | Why |
|-----------|-----|
| Use 3–8 photos per person | More photos → more robust centroid embedding |
| Vary lighting (bright, dim, mixed) | Prevents over-fitting to one lighting condition |
| Include slight angle variations | Slightly left, right, tilted — improves robustness |
| Ensure face fills at least 25% of the image | SCRFD needs sufficient detail to detect and embed |
| Use the same camera/resolution as detection | Reduces domain shift between enrollment and inference |
| Avoid heavy filters or edits | Natural photos match live camera conditions better |

### How to delete a person

1. Go to `http://127.0.0.1:5000/create_encoding`
2. Enter the person's exact name in the Delete panel
3. Click **Delete Person**
4. Their folder is removed from `dataset/` and their entry is removed from `facenet_embeddings1.npy`

### What happens under the hood

```
Upload photos for "Alice"
    → saved to dataset/Alice/
    → background: face_app.get(img) for each photo
    → collect 512-d ArcFace embeddings from each photo
    → average all embeddings → "Alice"'s centroid
    → save to facenet_embeddings1.npy as {"Alice": centroid_vector}
    → detection stream picks it up within 30 seconds
```

---

## 19. Performance Characteristics

### Per-frame timing (approximate, CPU-only)

| Operation | Time |
|-----------|------|
| Camera frame capture | ~5–10ms |
| InsightFace SCRFD detection (320×320) | ~80–100ms |
| InsightFace ArcFace embedding (per face) | ~30–50ms |
| YOLO weapon detection | ~80–120ms |
| YOLOv8n person detection | ~40–60ms |
| OpenCV draw operations | ~1–2ms |
| JPEG encode | ~5ms |
| **Total per processed frame (face+weapon parallel)** | **~180–220ms** |

### Effective throughput

With `FRAME_SKIP=3` and a 30 fps camera:
- Raw frames: 30 fps
- Processed frames: 10 fps
- Detection latency: ~200ms (~2 processed frames behind real-time)

This means the system detects a threat within approximately 0.2 seconds of it appearing — well within the response time needed for security applications.

### Memory usage

| Component | Memory |
|-----------|--------|
| Python interpreter | ~50MB |
| PyTorch (YOLO models) | ~200–300MB |
| InsightFace (ONNX Runtime) | ~150–200MB |
| OpenCV | ~30MB |
| Flask | ~20MB |
| **Total at runtime** | **~450–600MB** |

### CPU usage

With `FRAME_SKIP=3` on a modern quad-core CPU: approximately 40–70% CPU utilisation on one core (the detection threads run on available cores). The FRAME_SKIP value is the primary lever for reducing CPU load.

---

## 20. Security Design

### Credential management

- All credentials (email sender, password, recipient) are read from environment variables
- The `.env` file is in `.gitignore` and is never committed to the repository
- An `.env.example` template is committed showing which variables to set
- Gmail App Password is used instead of account password — it can be revoked independently without changing your Google account password

### File upload security

```python
safe_person = secure_filename(person_name)        # path traversal prevention
if _allowed_file(img.filename):                    # extension allowlist
    img.save(os.path.join(folder, secure_filename(img.filename)))
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # DoS size limit
```

- `secure_filename()` prevents directory traversal attacks (e.g. `../../etc/passwd` as filename)
- Extension allowlist (jpg, jpeg, png) prevents executable upload
- 10MB size limit prevents memory exhaustion

### Face data privacy

- Face images and embeddings are stored locally only
- Neither face images nor embeddings are sent to any external service
- InsightFace models run entirely locally (ONNX Runtime, no cloud API)
- The `.gitignore` excludes `dataset/`, `facenet_embeddings1.npy`, and `screenshot.png` from version control

### Email security

- SMTP over SSL (port 465) — the entire connection is encrypted
- Gmail App Password instead of account password
- Credentials never appear in log output (the warning message says "credentials not configured", not what they are)

---

## 21. Technology Decisions

### Why InsightFace instead of Haar cascade + FaceNet?

| Aspect | Old stack | New stack |
|--------|----------|-----------|
| Face detector | OpenCV Haar cascade | InsightFace SCRFD-500M |
| Face recogniser | keras-facenet (128-d) | InsightFace ArcFace (512-d) |
| Rotation robustness | ±30° | ±90° |
| Embedding quality | Moderate (128-d FaceNet) | High (512-d ArcFace angular margin) |
| Model coherence | Two separate libraries | Single unified library |
| Embedding consistency | Broken (enroll ≠ infer model) | Guaranteed (same singleton) |

The original codebase used `deepface` (Facenet512) for enrollment and `keras-facenet` for inference — two completely different models in incompatible embedding spaces. This silently broke face recognition entirely. InsightFace unifies both operations under a single `FaceAnalysis` object.

### Why YOLOv8n instead of YOLOv5su?

YOLOv5su is an "updated" version of YOLOv5 with ~7.2M parameters. YOLOv8n is the nano variant of the newer YOLOv8 architecture with ~3.2M parameters. For person detection at surveillance distances, both achieve comparable accuracy. YOLOv8n is ~2× faster and ~2× smaller. For a system already loaded with InsightFace + custom YOLO, reducing person detection cost matters.

### Why Flask instead of FastAPI or Django?

SafeGuard is a single-user local security application, not a multi-user web service. Flask's simplicity and minimal boilerplate fits this use case. FastAPI would add async complexity without benefit (the stream is inherently blocking). Django would add ORM, admin, migrations — none of which are needed.

### Why MJPEG instead of WebRTC or HLS?

MJPEG (`multipart/x-mixed-replace`) is the simplest possible video streaming protocol:
- No JavaScript video player needed — plain `<img>` tag works
- No codec negotiation
- No manifest files
- Works in all browsers
- Latency is predictable (one JPEG per processed frame)

The tradeoff is bandwidth (JPEG is larger than H.264) and single-client limitation, but for a local network security monitor these are non-issues.

### Why python-dotenv instead of hardcoded config?

Hardcoded credentials in source code have caused real security incidents. Even in a private repository, credentials should never be in source control. The `.env` pattern (used by virtually every modern application) keeps secrets separate from code, allows different values per environment, and makes it obvious where to look when something doesn't work.

---

## 22. Troubleshooting Guide

### Camera issues

**Symptom:** Black stream in browser / "Cannot open camera" in logs  
**Cause:** Another Python process is holding the camera  
**Fix:** `taskkill /F /IM python.exe` then restart

**Symptom:** Wrong camera opens  
**Fix:** Try `CAMERA_INDEX=0`, `1`, `2` in `.env`. You can find the right index by running `python -c "import cv2; [print(i, cv2.VideoCapture(i).isOpened()) for i in range(4)]"`

**Symptom:** Stream shows green tint / wrong colours  
**Cause:** Camera outputs in a different colour format  
**Fix:** This is usually a driver issue — try updating camera drivers

### Face recognition issues

**Symptom:** Enrolled person shows as "Unknown"  
**Cause 1:** Stream loaded before enrollment completed (file didn't exist yet)  
**Fix:** Wait up to 30 seconds — the stream auto-reloads embeddings

**Cause 2:** Only one enrollment photo → poor centroid representation  
**Fix:** Upload more photos with lighting/angle variation

**Cause 3:** Embeddings file from old model (wrong embedding space)  
**Fix:** Delete `facenet_embeddings1.npy`, re-enrol via `/create_encoding`

**Cause 4:** Poor lighting during live detection  
**Fix:** Lower `FACE_THRESHOLD` to `0.25`–`0.28`

**Symptom:** Wrong person recognised  
**Fix:** Raise `FACE_THRESHOLD` to `0.42`–`0.45`

### Email issues

**Symptom:** No email received  
**Check 1:** Run `python scripts/check_env.py` — are all three variables set?  
**Check 2:** Is `ALERT_EMAIL_PASSWORD` an App Password (16 chars) or your account password?  
**Check 3:** Is 2-Step Verification enabled on the Gmail account? (Required for App Passwords)  
**Check 4:** Check spam folder

**Symptom:** "Failed to send alert email: [Errno 11001]"  
**Cause:** No internet connection  
**Fix:** Restore network connectivity

**Symptom:** "Email alert skipped — credentials not configured"  
**Fix:** Create `.env` file with all three variables

### Detection issues

**Symptom:** Weapons not detected  
**Fix:** Lower `WEAPON_THRESHOLD` to `0.45`. Ensure the weapon is clearly visible and not occluded.

**Symptom:** Many false weapon alarms  
**Fix:** Raise `WEAPON_THRESHOLD` to `0.70`–`0.80`

**Symptom:** Persons not detected  
**Fix:** Lower `PERSON_THRESHOLD` to `0.35`. Ensure adequate lighting.

**Symptom:** Alarm fires for a fraction of a second then stops  
**Cause:** Threat condition briefly true then false  
**Fix:** Raise `ALARM_EMAIL_THRESHOLD` to see if sustained — normal behaviour for brief detections

### Performance issues

**Symptom:** Stream is very laggy  
**Fix 1:** Increase `FRAME_SKIP` to `5` or `6`  
**Fix 2:** Close other CPU-intensive applications  
**Fix 3:** Reduce camera resolution (add `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)` to `_open_camera()`)

**Symptom:** High memory usage  
**Cause:** Normal — PyTorch and ONNX Runtime keep models in memory  
**Expected:** ~450–600MB is normal

---

## 23. Project Evolution & Upgrade History

### Original stack (v1)

| Component | Technology |
|-----------|-----------|
| Face detection | OpenCV Haar cascade (`haarcascade_frontalface_default.xml`) |
| Face recognition | `keras-facenet` (128-d FaceNet embeddings) |
| Enrollment embedding | `deepface` with Facenet512 model (512-d, un-normalised) |
| Person detection | YOLOv5su |
| Configuration | Hardcoded values in source code |
| Code structure | Detection logic duplicated in `main.py` and `app.py` |

**Critical bug in v1:** Enrollment used `deepface` Facenet512 (512-d, norm ≈ 15–30). Detection used `keras-facenet` FaceNet-128d (128-d). These are completely different embedding spaces and different dimensions — every face comparison was mathematically invalid. Recognition appeared to work (no crash) but produced random similarity scores. This bug silently made face recognition non-functional from day one.

### Phase 1 & 2 upgrades (current)

**Phase 1 — Detection model upgrades:**
- YOLOv5su → YOLOv8n for person detection (2× faster, same accuracy)
- Configuration extracted to `config.py` with environment variable backing
- Hardcoded email credentials removed

**Phase 2 — Face pipeline rewrite:**
- Haar cascade + FaceNet → InsightFace buffalo_s (SCRFD-500M + ArcFace)
- Enrollment and inference unified under single `FaceAnalysis` singleton
- Embedding dimension validation in `utils.py`
- `pipeline.py` created to eliminate ~120-line code duplication

**Phase 3 — Planned (future):**
- Unified single YOLO model for both person and weapon detection
- This would eliminate one model inference call per frame

### Bug fixes applied

| Bug | Root cause | Fix |
|-----|-----------|-----|
| Face always Unknown | utils.py norm check rejected valid InsightFace embeddings (norm ≈ 25, not ≈ 1.0) | Removed norm check; cosine sim handles arbitrary norms |
| App crash on headless server | pygame.mixer.init() called at import time | Lazy init — mixer starts on first alarm call |
| Email crash with None path | os.path.exists(None) raises TypeError | Added `if image_path and` guard |
| Wrong CSS path in index.html | `../static/styles.css` relative path (fragile) | Replaced with Flask `url_for()` |
| Error messages shown in green | Message colour hard-coded as green | Conditional Jinja2 class based on ❌/✅ prefix |
| CLI never updated enrolled faces | known_faces loaded once, never refreshed | 30-second periodic reload (same as web stream) |
| script.js never loaded | No template referenced it | Deleted dead file |
