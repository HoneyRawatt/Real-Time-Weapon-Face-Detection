# SafeGuard — Real-Time Weapon & Face Detection

A real-time security surveillance system that triggers an audio alarm and sends an email alert when it simultaneously detects:
- A **person** in the frame
- An **unknown face** (not enrolled in the safe-people database)
- A **weapon** (gun or knife)

Supports two run modes : a Flask web UI and a CLI window.

---

## Models

| Task | Model | File |
|------|-------|------|
| Weapon detection | Custom YOLO11s (100 epochs) | `best100.pt` |
| Person detection | YOLOv8n (COCO pretrained) | `yolov8n.pt` |
| Face detection | InsightFace SCRFD-500M | downloaded to `~/.insightface/` |
| Face recognition | InsightFace ArcFace w600k_mbf (512-d) | downloaded to `~/.insightface/` |

---

## Quick Start

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install PyTorch (CPU)

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

For NVIDIA GPU replace `cpu` with your CUDA version (e.g. `cu121`).

### 3. Install remaining dependencies

```powershell
pip install -r requirements.txt
```

InsightFace will download the `buffalo_s` model pack (~85 MB) to `~/.insightface/models/` on first run.

### 4. Configure email alerts

Copy `.env.example` to `.env` and fill in your values:

```
ALERT_SENDER_EMAIL=youremail@gmail.com
ALERT_EMAIL_PASSWORD=xxxx xxxx xxxx xxxx
ALERT_TO_EMAIL=recipient@gmail.com
```

`ALERT_EMAIL_PASSWORD` must be a **Gmail App Password** (not your account password).  
Generate one at: Google Account → Security → 2-Step Verification → App Passwords.

### 5. Run

**Web UI (recommended):**
```powershell
python app.py
# Open http://127.0.0.1:5000
```

**CLI (OpenCV window):**
```powershell
python main.py
# Press 'q' to quit
```

---

## Enrolling Safe People

1. Go to `http://127.0.0.1:5000/create_encoding`
2. Enter a name and upload 3–5 clear, well-lit face photos (JPG/PNG)
3. The system generates ArcFace embeddings in the background
4. Enrolled faces appear with a **green box** in the detection stream; unknown faces get a **red box**

---

## Web Routes

| Route | Purpose |
|-------|---------|
| `/` | Home page |
| `/detection` | Live MJPEG detection stream |
| `/safepeople` | List enrolled people |
| `/create_encoding` | Add or remove people |
| `/stop_stream` | Stop the active stream |

---

## Configuration

All tuneable values are in `config.py` and can be overridden via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CAMERA_INDEX` | `2` | OpenCV camera device index |
| `FACE_THRESHOLD` | `0.35` | ArcFace cosine similarity threshold |
| `WEAPON_THRESHOLD` | `0.6` | YOLO weapon confidence threshold |
| `PERSON_THRESHOLD` | `0.5` | YOLO person confidence threshold |
| `ALARM_COOLDOWN` | `5` | Seconds between repeated alarms |
| `ALARM_EMAIL_THRESHOLD` | `3` | Alarm triggers before email is sent |
| `FRAME_SKIP` | `3` | Process every Nth frame (higher = lighter CPU) |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Black camera screen | Another process is holding the camera. Run `taskkill /F /IM python.exe` then restart. |
| All faces show Unknown | Re-enrol via `/create_encoding`. Check Flask log for embedding errors. |
| Email not sending | Confirm `.env` exists with correct values. Use an App Password, not your account password. |
| Camera not found | Change `CAMERA_INDEX` in `.env` (try `0`, `1`, `2`). |
| Slow detection | Increase `FRAME_SKIP` in `.env` (e.g. `5`). |
