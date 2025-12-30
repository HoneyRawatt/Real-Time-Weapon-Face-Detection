# 🔍 WEAPON DETECTION SYSTEM - COMPREHENSIVE DOCUMENTATION

---

## 📋 PROJECT OVERVIEW

The **Weapon Detection System** is an intelligent real-time computer vision application that detects threats in video feeds by identifying:
- **Unknown/Suspicious Persons** (faces not in the safe database)
- **Weapons** (guns, knives)
- **People/Persons** in the scene

When all three threat indicators are detected simultaneously, the system:
1. **Triggers an alarm** (audio alert)
2. **Captures screenshot** of the threat
3. **Sends email alert** with the screenshot to security personnel

The application provides both:
- **CLI Mode** (`main.py`) - Terminal-based real-time detection
- **Web UI** (`app.py`) - Flask web interface for management and monitoring

---

## 🏗️ PROJECT ARCHITECTURE

```
Weapon_Detection/
├── app.py                          # Flask web UI server
├── main.py                         # CLI detection engine
├── face_detection.py               # Face recognition & embedding matching
├── weapon_detection.py             # YOLO-based weapon detection
├── person_detection.py             # YOLO-based person detection
├── email_sender.py                 # Email alert system
├── alarm.py                        # Audio alarm system
├── utils.py                        # Helper utilities
├── requirements.txt                # Python dependencies
├── best100.pt                      # YOLO weapon detection model (trained)
├── yolov5su.pt                     # YOLO person detection model (pretrained)
├── yolov8n.pt                      # Alternative YOLO model
├── yolo11n.pt                      # Alternative YOLO model
├── coco2.txt                       # Weapon class labels
├── haarcascade_frontalface_default.xml  # Face detection cascade
├── facenet_embeddings1.npy         # Known faces database (embeddings)
├── alarm.wav                       # Alarm sound file
├── dataset/                        # Folder for safe people's face images
├── templates/                      # HTML pages for web UI
│   ├── index.html                 # Home page
│   ├── detection.html             # Detection page
│   ├── safepeople.html            # Known people list
│   └── create_encoding.html       # Add/remove people
├── static/                         # CSS/JS for web UI
│   ├── styles.css
│   └── script.js
└── runs100epochs/                  # Training results (if model retrained)
```

---

## 🔧 TECH STACK & MODULES

### **1. CORE DEPENDENCIES**

| Module | Version | Purpose | Why Used |
|--------|---------|---------|----------|
| **numpy** | >=1.24.0 | Numerical computing | Fast array operations for embeddings & image processing |
| **opencv-python** | >=4.7.0 | Computer vision library | Video capture, frame processing, drawing boxes |
| **ultralytics** | >=8.0.0 | YOLO framework | State-of-art object detection (weapons, persons) |
| **keras-facenet** | ==0.3.2 | Face embedding model | Generates 128-dim embeddings for face recognition |
| **scikit-learn** | >=1.2.0 | ML utilities | Cosine similarity for face matching |
| **deepface** | >=0.0.75 | Face detection & analysis | Robust face detection in frames |
| **flask** | >=2.0.0 | Web framework | Creates HTTP API and web UI |
| **pygame** | >=2.1.0 | Game/audio library | Plays alarm sound when threat detected |
| **torch & torchvision** | 2.9.1+cpu | Deep learning backend | Powers YOLO inference (CPU or GPU) |
| **tensorflow** | 2.20.0 | Deep learning framework | Supports Keras-FaceNet embeddings |

### **2. PYTHON ENVIRONMENT**

```
Environment: Python 3.11 (Windows)
Location: .venv311/
Virtual Env Manager: venv
GPU Support: Optional (CPU by default)
```

**Why Python 3.11?**
- Many older libraries (`keras-facenet`, `ultralytics`) require Python ≤ 3.11
- Python 3.13 breaks compatibility with these packages
- Ensures all dependencies resolve without conflicts

---

## 📦 DETAILED MODULE BREAKDOWN

### **1. main.py** - Detection Engine (CLI)
**Purpose:** Real-time threat detection from camera feed

**Key Functions:**

```python
detect_objects_in_realtime()  # Main loop
├─ Input: Camera feed (index 2)
├─ Processing:
│  ├─ Face detection (parallel thread)
│  ├─ Weapon detection (parallel thread)
│  └─ Person detection (main thread)
├─ Logic: IF person AND unknown_face AND weapon THEN alarm
└─ Output: Display frame, trigger alarm, send email

generate_frames(stop_event, streaming_flag)  # For Flask
├─ Same detection pipeline
├─ Output: MJPEG video stream (yields frames)
└─ Used by: app.py for web UI video feed

capture_and_send_email(frame)
├─ Saves frame to "screenshot.png"
├─ Calls send_email_with_attachment()
└─ Triggered: When alarm_count > 2
```

**Detection Logic:**
- Runs at 480x640 resolution
- Skips frames for speed (FRAME_SKIP=1)
- Uses threading for parallel face + weapon detection
- Triggers alarm when all 3 conditions met:
  1. Person detected
  2. Unknown face detected (confidence < 0.6)
  3. Weapon detected (gun or knife, confidence > 0.6)

**Alarm Cooldown:**
- ALARM_COOLDOWN = 5 seconds (prevent spam)
- alarm_count increments each trigger
- Email sent only after alarm_count > 2 (waits for 3 detections)

---

### **2. face_detection.py** - Face Recognition
**Purpose:** Detect unknown faces using embeddings

**Technology Stack:**
- **Cascade Detector:** `haarcascade_frontalface_default.xml` (Haar cascade)
- **Embedding Model:** `keras-facenet` (128-dimensional embeddings)
- **Similarity Metric:** Cosine similarity

**Functions:**

```python
get_embedding(face_img)  → 128-dim vector
├─ Input: Face image cropped to 160x160
├─ Model: FaceNet512 (from keras-facenet)
└─ Output: 128-dimensional embedding vector

detect_faces(frame, known_faces, threshold)  → boolean
├─ Known faces: Dictionary {name: embedding_vector}
├─ Threshold: 0.6 (confidence score)
├─ Process:
│  1. Detect all faces in frame (Haar cascade)
│  2. Extract embedding for each face
│  3. Compare to known faces (cosine similarity)
│  4. If best_score < 0.6 → mark as "Unknown"
└─ Returns: True if unknown face detected
```

**How Face Recognition Works:**
1. **Training Phase:** Admin adds known people's photos to `dataset/`
2. **Encoding Phase:** App calculates embeddings, stores in `facenet_embeddings1.npy`
3. **Detection Phase:** New faces compared to stored embeddings
4. **Threshold:** Cosine similarity > 0.6 = "Known", < 0.6 = "Unknown"

**Why FaceNet?**
- Produces consistent embeddings for same person
- Robust to lighting, angles, expressions
- Industry standard for face recognition

---

### **3. weapon_detection.py** - Weapon Detection
**Purpose:** Detect guns and knives using trained YOLO model

**Model:**
- **Model File:** `best100.pt` (custom trained YOLO)
- **Architecture:** YOLOv8 (trained for 100 epochs)
- **Classes:** Defined in `coco2.txt`

**Functions:**

```python
detect_weapons(frame)  → boolean
├─ Input: Frame from camera
├─ Model: YOLO('best100.pt')
├─ Process:
│  1. Run inference on frame
│  2. Filter detections (confidence > 0.6)
│  3. Check if class is "gun" or "knife"
│  4. Draw bounding boxes on frame
├─ Returns: True if weapon detected
```

**Detection Logic:**
- Confidence threshold: 0.6 (60% certainty)
- Only marks "gun" or "knife" detections
- Draws RED bounding box around weapons
- Returns True if any weapon found

**Why Custom Model (`best100.pt`)?**
- Trained on weapon-specific dataset
- 100 epochs = well-trained (good accuracy)
- Better than pretrained COCO models (which don't focus on weapons)

---

### **4. person_detection.py** - Person Detection
**Purpose:** Detect people/persons in the scene

**Model:**
- **Model File:** `yolov5su.pt` (pretrained YOLOv5)
- **Trained On:** COCO dataset (80 classes including "person")

**Functions:**

```python
detect_people(frame)  → list of boxes
├─ Input: Frame from camera
├─ Model: YOLO('yolov5su.pt')
├─ Process:
│  1. Run inference
│  2. Filter by class == 0 (person class)
│  3. Filter by confidence > 0.5
│  4. Return bounding box coordinates
├─ Returns: List of (x1, y1, x2, y2) tuples
```

**Output:**
- Each tuple: (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
- Used by: `main.py` to check if person exists

**Why YOLOv5?**
- Fast inference speed (real-time capable)
- Pretrained on COCO (excellent person detection)
- Sufficient accuracy for presence detection

---

### **5. app.py** - Flask Web UI
**Purpose:** Web interface for monitoring and managing system

**Architecture:**
- **Framework:** Flask (Python web framework)
- **Port:** 127.0.0.1:5000
- **Video Streaming:** MJPEG format

**Endpoints:**

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Home page |
| `/detection` | GET | Live detection page |
| `/video_feed` | GET | MJPEG video stream |
| `/stop_stream` | GET | Stop streaming |
| `/safepeople` | GET | List known people |
| `/create_encoding` | GET/POST | Add/remove people |

**Key Features:**

```python
@app.route('/video_feed')
├─ Returns: MJPEG stream (real-time video)
├─ Source: generate_frames() from main.py
└─ Continuous frame updates

@app.route('/create_encoding')
├─ POST action="add": Upload images for new person
│  ├─ Save images to dataset/{person_name}/
│  └─ Recalculate embeddings
├─ POST action="delete": Remove person from database
│  ├─ Delete dataset/{person_name}/ folder
│  └─ Update embeddings file
└─ GET: Show form

def update_embeddings()
├─ Scans dataset/ folder
├─ Loads all images
├─ Generates embeddings (DeepFace.represent)
├─ Averages embeddings per person
└─ Saves to facenet_embeddings1.npy

@app.route('/safepeople')
├─ Lists all known people
├─ Shows thumbnail image
└─ For management interface
```

**Data Flow:**
```
User uploads image → Flask receives → Save to dataset/
→ update_embeddings() → DeepFace generates embedding
→ Save to facenet_embeddings1.npy → Face detection uses it
```

---

### **6. email_sender.py** - Email Alerts
**Purpose:** Send threat notifications via email

**Configuration (Environment Variables):**
```
ALERT_SENDER_EMAIL = "memersasta20@gmail.com"
ALERT_EMAIL_PASSWORD = "@bcdefghijklmnop"  (16-char app password)
ALERT_TO_EMAIL = "rawathoney952@gmail.com"
```

**Functions:**

```python
send_email_with_attachment(image_path, to_email=None)
├─ Reads credentials from environment
├─ Creates MIME message with:
│  ├─ Subject: "Security Alert: Unknown Person Detected"
│  ├─ Body: Alert description
│  └─ Attachment: screenshot.png
├─ Connects to: smtp.gmail.com:465 (TLS)
├─ Sends via: Gmail SMTP
└─ Logs: Success/failure with emojis
```

**SMTP Configuration:**
- **Server:** smtp.gmail.com
- **Port:** 465 (SSL/TLS)
- **Authentication:** Gmail app password (not account password)
- **Reason:** Gmail requires app-specific password for programmatic access

**Why Environment Variables?**
- Credentials NOT in code (security)
- Can be updated without code changes
- Follows 12-factor app principles

---

### **7. alarm.py** - Audio Alerts
**Purpose:** Play alarm sound when threat detected

**Technology:**
- **Library:** pygame.mixer
- **Sound File:** alarm.wav

**Functions:**

```python
start_alarm()
├─ Plays alarm.wav sound
└─ Max duration: 5000ms (5 seconds)

stop_alarm()
├─ Stops sound immediately
└─ Called when threat condition clears
```

**Flow:**
```
Threat detected → start_alarm() (threading) → User hears alarm
Threat cleared → stop_alarm() → Silence
```

**Why Pygame?**
- Cross-platform (Windows, Linux, Mac)
- Supports WAV files
- Non-blocking audio (doesn't freeze video)

---

### **8. utils.py** - Helpers
**Purpose:** Utility functions

**Functions:**

```python
load_known_faces(embeddings_file)
├─ Loads facenet_embeddings1.npy
├─ Returns: Dictionary {person_name: embedding_vector}
└─ Used by: face_detection.py for matching
```

---

## 🚀 WORKFLOW & DATA FLOW

### **Phase 1: Setup (One-time)**

```
1. User creates Python 3.11 venv
   ↓
2. User installs PyTorch (CPU)
   ↓
3. User installs all dependencies (requirements.txt)
   ↓
4. User uploads photos of safe people via Flask web UI
   ↓
5. System generates embeddings → saved to facenet_embeddings1.npy
   ↓
6. System ready for detection
```

### **Phase 2: Real-time Detection (Continuous)**

```
Camera Input (30 fps)
    ↓
Frame received
    ↓
─────────────────────────────────────────────
│ PARALLEL PROCESSING                       │
│ ┌─────────────────┐   ┌──────────────────┐│
│ │ Thread 1:       │   │ Thread 2:        ││
│ │ Face Detection  │   │ Weapon Detection ││
│ │ (keras-facenet) │   │ (YOLO best100)   ││
│ └──────┬──────────┘   └────────┬─────────┘│
│        │                       │          │
│ └──────────────────────────────┘          │
└─────────────────────────────────────────────
    ↓
Person Detection (yolov5su)
    ↓
─────────────────────────────────────────────
│ THREAT ANALYSIS                           │
│ IF (person AND unknown_face AND weapon)   │
│    THEN alarm_condition = True            │
│ ELSE alarm_condition = False              │
└─────────────────────────────────────────────
    ↓
IF alarm_condition:
  ├─ alarm_count++
  ├─ Start alarm (pygame) via thread
  ├─ IF alarm_count > 2:
  │  ├─ Capture frame → screenshot.png
  │  └─ Send email with attachment
  └─ Render bounding boxes
    ↓
Display frame on screen / Stream to Flask
    ↓
GOTO: Frame received
```

### **Phase 3: Web UI Management**

```
User accesses http://127.0.0.1:5000
    ↓
├─ View home page
├─ Watch live detection stream (/video_feed)
├─ View list of safe people (/safepeople)
├─ Add new safe person (/create_encoding)
│  ├─ Upload images
│  ├─ System generates embeddings
│  └─ Stored in facenet_embeddings1.npy
└─ Delete person from database
```

---

## 📊 COMPARISON: CLI vs WEB UI

| Feature | CLI (`main.py`) | Web UI (`app.py`) |
|---------|-----------------|------------------|
| **Display** | OpenCV window | Browser window |
| **Video Feed** | Local only | Accessible remotely |
| **Management** | Manual file editing | Web forms |
| **Add People** | Manual folder creation | Upload via UI |
| **Alarms** | Local speaker | Local speaker (configurable) |
| **Email** | Automatic | Automatic |
| **Use Case** | Testing, development | Production deployment |
| **Remote Access** | No | Yes (if exposed) |

---

## 🔐 SECURITY FEATURES

### **1. Credential Management**
- ❌ NOT hardcoded in source
- ✅ Environment variables (ALERT_SENDER_EMAIL, etc.)
- ✅ `.env.example` template provided
- ✅ `.gitignore` excludes `.env` and `__pycache__`

### **2. Face Database**
- Embeddings stored in `.npy` file (not human-readable)
- Face images kept locally in `dataset/` folder
- Can delete people from system anytime

### **3. Email Authentication**
- Gmail app-specific password (16 characters)
- NOT the account password
- Revokable from Google Account settings

### **4. File Permissions**
- `facenet_embeddings1.npy` → read-only in production
- `alarm.wav` → read-only
- Model files (`.pt`) → read-only

---

## 🎯 WHY EACH TECHNOLOGY WAS CHOSEN

| Technology | Alternative | Why Chosen |
|-----------|-------------|-----------|
| **OpenCV** | scikit-image | Fastest, most optimized for real-time video |
| **YOLO** | R-CNN, SSD | Fastest inference, best for real-time detection |
| **FaceNet** | VGGFace, ArcFace | Best embeddings quality, widely used |
| **Flask** | Django, FastAPI | Lightweight, perfect for simple web UI |
| **PyGame** | pydub | Simple audio playback, cross-platform |
| **Gmail SMTP** | AWS SES, Twilio | Free, doesn't require API setup, familiar |
| **Keras-FaceNet** | Tensorflow-Hub | Directly installable via pip, no model download |
| **Python 3.11** | 3.10, 3.13 | Best compatibility with all libraries |

---

## 🛠️ HOW TO CREATE THIS PROJECT FROM SCRATCH

### **Step 1: Environment Setup**
```powershell
# Create Python 3.11 venv
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel
```

### **Step 2: Install Core Dependencies**
```powershell
# Install CPU PyTorch
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision --upgrade

# Install requirements
pip install -r requirements.txt
```

### **Step 3: Download/Prepare Models**
```
✓ best100.pt (custom trained YOLO - provided)
✓ yolov5su.pt (pretrained - auto-downloaded by ultralytics)
✓ yolov8n.pt (pretrained - auto-downloaded by ultralytics)
✓ haarcascade_frontalface_default.xml (built-in to OpenCV)
```

### **Step 4: Prepare Data**
```
Create dataset/ folder
├─ person1/
│  ├─ face1.jpg
│  ├─ face2.jpg
│  └─ face3.jpg
└─ person2/
   ├─ face1.jpg
   └─ face2.jpg
```

### **Step 5: Generate Face Embeddings**
```
Access http://127.0.0.1:5000/create_encoding
├─ Upload images for each person
├─ System auto-generates embeddings
└─ Stored in facenet_embeddings1.npy
```

### **Step 6: Configure Email**
```powershell
$env:ALERT_SENDER_EMAIL = 'youremail@gmail.com'
$env:ALERT_EMAIL_PASSWORD = 'xxxx xxxx xxxx xxxx'  # App password
$env:ALERT_TO_EMAIL = 'recipient@gmail.com'
```

### **Step 7: Run System**
```powershell
# Option A: CLI Detection
python main.py

# Option B: Web UI + Detection
python app.py
# Then visit http://127.0.0.1:5000
```

---

## 📈 PERFORMANCE CHARACTERISTICS

### **Frame Processing**
- **Resolution:** 480×640 pixels
- **FPS Target:** ~10 fps (due to threading)
- **Bottleneck:** Face embedding generation (slowest)

### **Model Inference Times**
| Model | Time | Purpose |
|-------|------|---------|
| Person Detection (YOLO) | ~50ms | Detect people |
| Face Detection (Haar) | ~30ms | Find faces |
| Face Embedding (FaceNet) | ~150ms | Generate 128-dim vector |
| Weapon Detection (YOLO) | ~100ms | Detect guns/knives |
| **Total per frame** | ~330ms | Sequential (or parallel where possible) |

### **Memory Usage**
- Base: ~200MB (OpenCV + PyTorch)
- Per frame: +10MB (temporary)
- Embeddings file: ~1MB (for 50 people)

---

## ✅ VERIFICATION CHECKLIST

- ✅ All dependencies installed
- ✅ Models downloaded (best100.pt, yolov5su.pt)
- ✅ Camera connected (device index 2)
- ✅ Face embeddings generated
- ✅ Email credentials configured
- ✅ Alarm sound file present
- ✅ Flask app accessible
- ✅ Real-time detection running

---

## 🐛 TROUBLESHOOTING

| Issue | Cause | Solution |
|-------|-------|----------|
| "Module not found" | Wrong Python version | Use Python 3.11 |
| "Camera not found" | Wrong device index | Change `cv2.VideoCapture(2)` to `cv2.VideoCapture(0)` |
| "Face detection slow" | Model warming up | First run is slower (model loads) |
| "Email not sending" | App password wrong | Generate new password from Google Account |
| "CUDA errors" | GPU PyTorch on CPU system | Reinstall CPU PyTorch |

---

## 📞 SUMMARY

This **Weapon Detection System** combines multiple computer vision technologies into an integrated real-time threat detection platform. It uses:

1. **YOLO** for fast object detection (people, weapons)
2. **FaceNet** for face recognition and anomaly detection
3. **Flask** for web-based management
4. **Gmail SMTP** for alert notifications
5. **PyGame** for audio alerts

The system is modular, allowing each component to be tested independently, and scalable for deployment in various security scenarios.

