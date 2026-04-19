"""
SafeGuard — Flask web interface.

Routes
------
GET  /                  Home page
GET  /detection         Live stream viewer
GET  /video_feed        MJPEG stream (consumed by <img> in detection.html)
GET  /stop_stream       Stop the running stream
GET  /safepeople        List enrolled people
GET/POST /create_encoding  Enrol or delete people

Security hardening applied
--------------------------
• werkzeug.utils.secure_filename() on every upload path
• Extension allowlist — only .jpg / .jpeg / .png accepted
• MAX_CONTENT_LENGTH = 10 MB cap on incoming requests
• Person name sanitised before use as a directory name
• Embedding update runs in a daemon background thread so the HTTP response
  returns immediately instead of blocking for 30–120 seconds
"""
import os
import cv2
import numpy as np
import shutil
import threading
import logging

from flask import Flask, request, render_template, Response
from werkzeug.utils import secure_filename

from pipeline import generate_frames
from face_detection import get_face_app
from utils import load_known_faces
from config import EMBEDDINGS_FILE, DATASET_FOLDER

logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = DATASET_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB max upload

os.makedirs(DATASET_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# ── Stream state ──────────────────────────────────────────────────────────────

streaming  = True
stop_event = threading.Event()

# ── Embedding update state ────────────────────────────────────────────────────

_embed_running = False
_embed_lock    = threading.Lock()   # protects .npy file reads/writes


# ── Helpers ───────────────────────────────────────────────────────────────────

def _allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def _update_embeddings() -> None:
    """
    Generate ArcFace embeddings for every enrolled person and persist them.

    Uses the same InsightFace FaceAnalysis singleton as the detection path
    so enrollment and inference always use the same model and produce
    identically-dimensioned (512-d) vectors.
    """
    face_app = get_face_app()
    database: dict = {}

    # Load existing embeddings so people not in this batch are preserved.
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            database = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
        except Exception as exc:
            logger.warning(f"Could not load existing embeddings: {exc} — starting fresh.")
            database = {}

    for person in os.listdir(DATASET_FOLDER):
        person_path = os.path.join(DATASET_FOLDER, person)
        if not os.path.isdir(person_path):
            continue

        embeddings = []
        for img_name in os.listdir(person_path):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                logger.warning(f"  Cannot read '{img_path}' — skipping.")
                continue
            faces = face_app.get(img)
            if faces:
                embeddings.append(faces[0].embedding)   # 512-d ArcFace vector
                logger.debug(f"  Embedded '{img_name}' for '{person}'.")
            else:
                logger.warning(f"  No face detected in '{img_name}' — skipping.")

        if embeddings:
            database[person] = np.mean(embeddings, axis=0)
            logger.info(f"Enrolled '{person}' from {len(embeddings)} image(s).")
        else:
            logger.warning(f"No usable images for '{person}' — not enrolled.")

    with _embed_lock:
        np.save(EMBEDDINGS_FILE, database)
    logger.info(f"Embeddings saved.  Enrolled people: {list(database.keys())}")


def _run_update_embeddings_bg() -> None:
    """Background-thread wrapper — sets/clears the running flag."""
    global _embed_running
    _embed_running = True
    try:
        _update_embeddings()
    except Exception:
        logger.exception("Background embedding update failed.")
    finally:
        _embed_running = False


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detection")
def detection():
    return render_template("detection.html")


@app.route("/safepeople")
def safepeople():
    people = []
    for name in os.listdir(DATASET_FOLDER):
        person_folder = os.path.join(DATASET_FOLDER, name)
        if os.path.isdir(person_folder):
            images = [
                f for f in os.listdir(person_folder)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if images:
                people.append({
                    "name":  name,
                    "image": os.path.join(person_folder, images[0]),
                })
    return render_template("safepeople.html", people=people)


@app.route("/create_encoding", methods=["GET", "POST"])
def create_encoding():
    if request.method != "POST":
        return render_template("create_encoding.html")

    action = request.form.get("action")

    # ── Add person ────────────────────────────────────────────────────────────
    if action == "add":
        try:
            person_name = request.form.get("person_name", "").strip()
            if not person_name:
                return render_template("create_encoding.html",
                                       message="❌ Name cannot be empty.")

            images = request.files.getlist("images")
            if not images or all(img.filename == "" for img in images):
                return render_template("create_encoding.html",
                                       message="❌ No images uploaded.")

            # Use secure_filename on both the person name (folder) and each image.
            safe_person  = secure_filename(person_name)
            person_folder = os.path.join(DATASET_FOLDER, safe_person)
            os.makedirs(person_folder, exist_ok=True)

            saved = 0
            for img in images:
                if img and _allowed_file(img.filename):
                    img.save(os.path.join(person_folder, secure_filename(img.filename)))
                    saved += 1

            if saved == 0:
                return render_template("create_encoding.html",
                                       message="❌ No valid JPG/PNG files in upload.")

            # Fire-and-forget background embedding update.
            if not _embed_running:
                threading.Thread(target=_run_update_embeddings_bg, daemon=True).start()
                embed_msg = "Embedding generation started in background."
            else:
                embed_msg = "⏳ Embedding update already running — new images will be picked up."

            logger.info(f"Added {saved} image(s) for '{safe_person}'.")
            return render_template(
                "create_encoding.html",
                message=f"✅ '{person_name}' added ({saved} image(s)). {embed_msg}",
            )

        except Exception:
            logger.exception("Error adding person")
            return render_template("create_encoding.html",
                                   message="❌ Unexpected error — check server logs.")

    # ── Delete person ─────────────────────────────────────────────────────────
    elif action == "delete":
        try:
            person_name   = request.form.get("person_name", "").strip()
            safe_person   = secure_filename(person_name)
            person_folder = os.path.join(DATASET_FOLDER, safe_person)

            if not os.path.exists(person_folder):
                return render_template("create_encoding.html",
                                       message=f"❌ '{person_name}' not found.")

            shutil.rmtree(person_folder)

            # Remove the entry from the embeddings file atomically.
            if os.path.exists(EMBEDDINGS_FILE):
                with _embed_lock:
                    db = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
                    if person_name in db:
                        del db[person_name]
                    if safe_person in db and safe_person != person_name:
                        del db[safe_person]
                    np.save(EMBEDDINGS_FILE, db)

            logger.info(f"Deleted '{person_name}'.")
            return render_template("create_encoding.html",
                                   message=f"✅ '{person_name}' deleted.")

        except Exception:
            logger.exception("Error deleting person")
            return render_template("create_encoding.html",
                                   message="❌ Unexpected error — check server logs.")

    return render_template("create_encoding.html", message="❌ Unknown action.")


@app.route("/video_feed")
def video_feed():
    global streaming, stop_event
    stop_event.clear()
    streaming = True
    return Response(
        generate_frames(stop_event, lambda: streaming),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stop_stream")
def stop_stream():
    global streaming, stop_event
    streaming = False
    stop_event.set()
    logger.info("Stream stopped by user request.")
    return "Streaming stopped"


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
