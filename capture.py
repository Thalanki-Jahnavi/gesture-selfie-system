import cv2
import time
import numpy as np
import mediapipe as mp
import os
import psutil

SAVE_FOLDER = "selfies"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# Face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

countdown_active = False
countdown_start = 0
current_gesture = ""

frame_times = []


def capture_selfie(frame, gesture):
    filename = (
        f"{SAVE_FOLDER}/selfie_{gesture}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
    )
    cv2.imwrite(filename, frame)
    print("Saved:", filename)


def is_face_centered(faces, frame_shape):
    h, w, _ = frame_shape
    cx, cy = w // 2, h // 2

    for x, y, fw, fh in faces:
        face_cx = x + fw // 2
        face_cy = y + fh // 2

        if abs(face_cx - cx) < 80 and abs(face_cy - cy) < 80:
            return True
    return False


def detect_hand_gesture(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    tips = [
        hand_landmarks.landmark[i]
        for i in [
            mp_hands.HandLandmark.THUMB_TIP,
            mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP,
        ]
    ]

    thumb_tip = tips[0]
    thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

    index_tip, middle_tip, ring_tip, pinky_tip = tips[1:]

    thumb_up = (
        thumb_tip.y < thumb_ip.y < thumb_mcp.y
        and thumb_tip.y < wrist.y
        and all(tip.y > wrist.y for tip in [index_tip, middle_tip, ring_tip, pinky_tip])
    )

    peace = (
        index_tip.y < wrist.y
        and middle_tip.y < wrist.y
        and all(tip.y > wrist.y for tip in [thumb_tip, ring_tip, pinky_tip])
    )

    palm = all(tip.y < wrist.y for tip in tips)

    if thumb_up:
        return "thumbs_up"
    elif peace:
        return "peace"
    elif palm:
        return "palm"
    return None


while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    clean_frame = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Draw center box
    h, w, _ = frame.shape
    cv2.rectangle(
        frame, (w // 2 - 80, h // 2 - 80), (w // 2 + 80, h // 2 + 80), (255, 255, 0), 2
    )

    face_ok = is_face_centered(faces, frame.shape)

    if not face_ok:
        cv2.putText(
            frame,
            "Align Face in Box",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    # Hand detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = detect_hand_gesture(hand_landmarks)

            if gesture and not countdown_active and face_ok:
                current_gesture = gesture
                print(gesture, "detected!")
                countdown_active = True
                countdown_start = time.time()

    # Countdown
    if countdown_active:
        elapsed = time.time() - countdown_start
        remaining = 3 - int(elapsed)

        if remaining > 0:
            cv2.putText(
                frame,
                str(remaining),
                (250, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,
                (0, 0, 255),
                5,
            )
        else:
            capture_selfie(clean_frame, current_gesture)
            countdown_active = False

    # FPS
    frame_time = time.time() - start
    frame_times.append(frame_time)
    avg_fps = 1 / np.mean(frame_times[-30:]) if frame_times else 0

    mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2

    cv2.putText(
        frame,
        f"FPS: {avg_fps:.2f} | RAM: {mem:.1f}MB",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    cv2.imshow("Smart Selfie System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
