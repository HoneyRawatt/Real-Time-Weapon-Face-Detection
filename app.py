import os
import numpy as np
import shutil
from flask import Flask, request, render_template, Response, url_for
from deepface import DeepFace
from main import generate_frames
import threading

streaming = True
stop_event = threading.Event()

app = Flask(__name__)
UPLOAD_FOLDER = "dataset"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

EMBEDDINGS_FILE = "facenet_embeddings1.npy"

# ========== Home Page ==========
@app.route("/")
def home():
    return render_template("index.html")


# ========== Detection Page ==========
@app.route("/detection")
def detection():
    return render_template("detection.html")


# ========== Safe People Page ==========
@app.route("/safepeople")
def safepeople():
    people = []
    for name in os.listdir(UPLOAD_FOLDER):
        person_folder = os.path.join(UPLOAD_FOLDER, name)
        if os.path.isdir(person_folder):
            image_files = [f for f in os.listdir(person_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                image_path = os.path.join(person_folder, image_files[0])
                people.append({"name": name, "image": image_path})
    return render_template("safepeople.html", people=people)


# ========== Encoding Page ==========
@app.route("/create_encoding", methods=["GET", "POST"])
def create_encoding():
    if request.method == "POST":
        action = request.form.get("action")

        if action == "add":
            try:
                person_name = request.form["person_name"]
                images = request.files.getlist("images")

                person_folder = os.path.join(UPLOAD_FOLDER, person_name)
                os.makedirs(person_folder, exist_ok=True)

                for img in images:
                    img_path = os.path.join(person_folder, img.filename)
                    img.save(img_path)

                update_embeddings()

                return render_template("create_encoding.html", message=f"✅ {person_name} added successfully!")

            except Exception as e:
                return render_template("create_encoding.html", message=f"❌ Error: {str(e)}")

        elif action == "delete":
            try:
                person_name = request.form["person_name"]
                person_folder = os.path.join(UPLOAD_FOLDER, person_name)

                if not os.path.exists(person_folder):
                    return render_template("create_encoding.html", message=f"❌ {person_name} not found!")

                shutil.rmtree(person_folder)

                # Remove from embeddings
                if os.path.exists(EMBEDDINGS_FILE):
                    database = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()
                    if person_name in database:
                        del database[person_name]
                        np.save(EMBEDDINGS_FILE, database)

                return render_template("create_encoding.html", message=f"✅ {person_name} deleted successfully!")
            except Exception as e:
                return render_template("create_encoding.html", message=f"❌ Error: {str(e)}")

    return render_template("create_encoding.html")


# ========== Helper Function ==========
def update_embeddings():
    database = {}

    if os.path.exists(EMBEDDINGS_FILE):
        database = np.load(EMBEDDINGS_FILE, allow_pickle=True).item()

    for person in os.listdir(UPLOAD_FOLDER):
        person_path = os.path.join(UPLOAD_FOLDER, person)
        if os.path.isdir(person_path):
            embeddings = []
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    try:
                        emb = DeepFace.represent(img_path, model_name="Facenet512", enforce_detection=False)[0]["embedding"]
                        embeddings.append(emb)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
            if embeddings:
                database[person] = np.mean(embeddings, axis=0)

    np.save(EMBEDDINGS_FILE, database)
    print("✅ Face embeddings updated successfully!")


# ========== Video Feed ==========
@app.route('/video_feed')
def video_feed():
    global streaming, stop_event
    stop_event.clear()
    streaming = True
    return Response(generate_frames(stop_event, lambda: streaming),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



# ========== Stop Stream ==========
@app.route('/stop_stream')
def stop_stream():
    global streaming, stop_event
    streaming = False
    stop_event.set()
    return 'Streaming stopped'


if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
