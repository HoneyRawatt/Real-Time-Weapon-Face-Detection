import cv2
import numpy as np
import os
import sys
# Ensure project root is on sys.path so local modules import correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from face_detection import get_embedding
from utils import load_known_faces

EMBEDDINGS_FILE = 'facenet_embeddings1.npy'

known = {}
if os.path.exists(EMBEDDINGS_FILE):
    known = load_known_faces(EMBEDDINGS_FILE)
else:
    print('No embeddings file found; known faces DB is empty.')

cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print('Could not open camera index 2; trying index 0...')
    cap = cv2.VideoCapture(0)

ret, frame = cap.read()
cap.release()

if not ret:
    print('Failed to grab a frame from camera.')
    exit(1)

cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

if len(faces)==0:
    print('No faces detected in the captured frame.')
else:
    print(f'Detected {len(faces)} face(s).')

import math
from sklearn.metrics.pairwise import cosine_similarity

for i,(x,y,w,h) in enumerate(faces, start=1):
    face_img = frame[y:y+h, x:x+w]
    try:
        face_resized = cv2.resize(face_img, (160,160))
    except Exception as e:
        print('Skipping face: resize error', e)
        continue
    emb = get_embedding(face_resized)
    best_name = 'Unknown'
    best_score = -1.0
    for name, stored in known.items():
        score = cosine_similarity([emb], [stored])[0][0]
        if score > best_score:
            best_score = score
            best_name = name
    print(f'Face {i}: best_match={best_name}, score={best_score:.4f}')
    label = f'{best_name} ({best_score:.2f})'
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

out_path = 'face_test_result.png'
cv2.imwrite(out_path, frame)
print('Annotated frame saved to', out_path)