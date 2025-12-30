import cv2
import os
import time
import threading
from face_detection import detect_faces
from weapon_detection import detect_weapons
from alarm import start_alarm, stop_alarm
from utils import load_known_faces
from email_sender import send_email_with_attachment
from person_detection import detect_people  

CONFIDENCE_THRESHOLD = 0.6
ALARM_COOLDOWN = 5 
FRAME_SKIP = 1

def capture_and_send_email(frame):
    image_path = "screenshot.png"
    cv2.imwrite(image_path, frame)
    print(" Screenshot captured. Sending email...")
    send_email_with_attachment(image_path)

def is_inside(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2

def detect_objects_in_realtime():
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_alarm_time = 0
    alarm_playing = False
    alarm_count = 0
    frame_count = 0

    known_faces = load_known_faces("facenet_embeddings1.npy")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        result = {}

        
        face_thread = threading.Thread(target=lambda: result.update({"face": detect_faces(frame, known_faces, CONFIDENCE_THRESHOLD)}))
        weapon_thread = threading.Thread(target=lambda: result.update({"weapon": detect_weapons(frame)}))

        face_thread.start()
        weapon_thread.start()

   
        person_boxes = detect_people(frame)
        person_detected = len(person_boxes) > 0

        face_thread.join()
        weapon_thread.join()

        unknown_detected = result.get("face", False)
        weapon_detected = result.get("weapon", False)

       
        for (x1, y1, x2, y2) in person_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        print(f"Person: {person_detected}, Unknown: {unknown_detected}, Weapon: {weapon_detected}")

        if person_detected and unknown_detected and weapon_detected:
            current_time = time.time()
            if not alarm_playing or (current_time - last_alarm_time > ALARM_COOLDOWN):
                print(" Starting alarm!")
                threading.Thread(target=start_alarm).start()
                alarm_playing = True
                last_alarm_time = current_time
                alarm_count += 1

                if alarm_count > 2:
                    threading.Thread(target=capture_and_send_email, args=(frame,)).start()

        elif alarm_playing and (not person_detected or not unknown_detected or not weapon_detected):
            print(" Stopping alarm!")
            stop_alarm()
            alarm_playing = False

        cv2.imshow(" Real-time Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()


def generate_frames(stop_event, streaming_flag):
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_alarm_time = 0
    alarm_playing = False
    alarm_count = 0
    frame_count = 0

    known_faces = load_known_faces("facenet_embeddings1.npy")

    while streaming_flag() and not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        result = {}

        face_thread = threading.Thread(target=lambda: result.update({"face": detect_faces(frame, known_faces, CONFIDENCE_THRESHOLD)}))
        weapon_thread = threading.Thread(target=lambda: result.update({"weapon": detect_weapons(frame)}))
        face_thread.start()
        weapon_thread.start()

        person_boxes = detect_people(frame)
        person_detected = len(person_boxes) > 0

        face_thread.join()
        weapon_thread.join()

        unknown_detected = result.get("face", False)
        weapon_detected = result.get("weapon", False)

        for (x1, y1, x2, y2) in person_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        print(f"Person: {person_detected}, Unknown: {unknown_detected}, Weapon: {weapon_detected}")

        current_time = time.time()
        if person_detected and unknown_detected and weapon_detected:
            if not alarm_playing or (current_time - last_alarm_time > ALARM_COOLDOWN):
                print(" Starting alarm!")
                threading.Thread(target=start_alarm).start()
                alarm_playing = True
                last_alarm_time = current_time
                alarm_count += 1

                if alarm_count > 2:
                    threading.Thread(target=capture_and_send_email, args=(frame,)).start()
        elif alarm_playing and (not person_detected or not unknown_detected or not weapon_detected):
            print(" Stopping alarm!")
            stop_alarm()
            alarm_playing = False

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    stop_alarm()


if __name__ == "__main__":
    detect_objects_in_realtime()
