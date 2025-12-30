import cv2
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
embedder = FaceNet()

def get_embedding(face_img):
    face_img = face_img.astype("float32")
    face_img = np.expand_dims(face_img, axis=0)
    return embedder.embeddings(face_img)[0]

def detect_faces(frame, known_faces, threshold):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    unknown_detected = False
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))

        face_embedding = get_embedding(face)
        best_match = "Unknown"
        best_score = 0

        for name, stored_embedding in known_faces.items():
            score = cosine_similarity([face_embedding], [stored_embedding])[0][0]
            if score > best_score:
                best_score = score
                best_match = name

        if best_score < threshold:
            best_match = "Unknown"
            unknown_detected = True  

        cv2.putText(frame, f"{best_match} ({best_score:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    return unknown_detected
