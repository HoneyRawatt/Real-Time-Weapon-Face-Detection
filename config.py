"""
Central configuration for SafeGuard detection system.
All tuneable values live here, backed by environment variables so nothing
needs to be edited in source code between environments.

Copy .env.example → .env and fill in your values, or export the variables
directly in your shell / systemd unit.
"""
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "2"))

# ── Detection thresholds ──────────────────────────────────────────────────────
# ArcFace cosine similarity: same person ≈ 0.3–0.6, strangers ≈ 0.0–0.2
FACE_THRESHOLD   = float(os.getenv("FACE_THRESHOLD",   "0.35"))
WEAPON_THRESHOLD = float(os.getenv("WEAPON_THRESHOLD", "0.6"))
PERSON_THRESHOLD = float(os.getenv("PERSON_THRESHOLD", "0.5"))

# ── Alarm & alerting ──────────────────────────────────────────────────────────
ALARM_COOLDOWN        = int(os.getenv("ALARM_COOLDOWN",        "5"))   # seconds
ALARM_EMAIL_THRESHOLD = int(os.getenv("ALARM_EMAIL_THRESHOLD", "3"))   # alarms before email

# ── Performance ───────────────────────────────────────────────────────────────
# Process every Nth frame.  1 = every frame (heavy), 3 = ~3× lighter on CPU.
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))

# ── Paths ─────────────────────────────────────────────────────────────────────
EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE", "facenet_embeddings1.npy")
DATASET_FOLDER  = os.getenv("DATASET_FOLDER",  "dataset")

# ── Email ─────────────────────────────────────────────────────────────────────
EMAIL_SENDER    = os.getenv("ALERT_SENDER_EMAIL")
EMAIL_PASSWORD  = os.getenv("ALERT_EMAIL_PASSWORD")
EMAIL_RECIPIENT = os.getenv("ALERT_TO_EMAIL")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
