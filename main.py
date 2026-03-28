import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture(0)

blink_counter = 0
blink_total = 0
last_blink_time = 0
cooldown = 1

EAR_THRESHOLD = 0.21
CONSEC_FRAMES = 3

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        mesh_points = np.array(
            [(p.x * frame.shape[1], p.y * frame.shape[0])
            for p in results.multi_face_landmarks[0].landmark]
        )

        left_eye = mesh_points[LEFT_EYE]
        right_eye = mesh_points[RIGHT_EYE]

        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)

        ear = (leftEAR + rightEAR) / 2

        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= CONSEC_FRAMES:
                blink_total += 1
                last_blink_time = time.time()
                print("Blink detected")

            blink_counter = 0

        current_time = time.time()

        if blink_total > 0 and current_time - last_blink_time > cooldown:

            if blink_total == 1:
                print("Next Slide")
                pyautogui.press("right")

            elif blink_total == 2:
                print("Previous Slide")
                pyautogui.press("left")

            blink_total = 0

    cv2.putText(frame, "Blink Control Active", (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Eye Blink Slide Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
