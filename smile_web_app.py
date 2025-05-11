import cv2
import streamlit as st
import numpy as np
import mediapipe as mp
import time
from datetime import datetime
from PIL import Image
import os

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=1,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.5)

# Output directory
output_dir = "captured_smiles"
os.makedirs(output_dir, exist_ok=True)

# Define motivational messages
messages = [
    "Your smile lights up the world!",
    "That smile is contagious!",
    "What a wonderful smile you have!",
    "You just made the day brighter!",
    "Keep smiling, it's beautiful!"
]

# Check if teeth or tongue is visible
TEETH_LANDMARKS = [13, 14]
TONGUE_LANDMARKS = [19, 87, 88]

# Smile detection logic
def detect_smile(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            teeth_visible = all(
                face_landmarks.landmark[i].z < -0.05 for i in TEETH_LANDMARKS
            )
            tongue_visible = all(
                face_landmarks.landmark[i].z < -0.05 for i in TONGUE_LANDMARKS
            )
            if teeth_visible or tongue_visible:
                return True, "Smile detected!"
    return False, "Please smile to continue."

# Streamlit UI
st.set_page_config(page_title="Smile Detector", layout="centered")
st.title("😊 Smile Detector App")
status_text = st.empty()
image_display = st.empty()

# Initialize webcam
cap = cv2.VideoCapture(0)
detected = False
start_time = time.time()

while cap.isOpened() and not detected:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    smile, msg = detect_smile(frame)
    status_text.markdown(f"### {msg}")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_display.image(rgb_frame, channels="RGB")

    if smile:
        detected = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(output_dir, f"smile_{timestamp}.jpg")
        cv2.imwrite(img_path, frame)
        status_text.markdown(f"## 😀 {np.random.choice(messages)}")
        image_display.image(rgb_frame, caption="Smile Captured!", channels="RGB")
        time.sleep(4)
        break

    if time.time() - start_time > 20:
        status_text.markdown("### Timeout. Please try again.")
        break

cap.release()
st.markdown("---")
st.markdown("#### Thank you for trying the Smile Detector!")
