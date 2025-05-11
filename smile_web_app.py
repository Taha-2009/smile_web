import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image
import os
from datetime import datetime

st.set_page_config(page_title="Smile Detector", layout="centered")
st.title("😊 Smile Detector")
st.markdown("---")

st.write("Allow camera access and smile to light up the world!")

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Create folder to save smile photos
output_folder = "smile_photos"
os.makedirs(output_folder, exist_ok=True)

# Helper function: detect smile (mouth open, teeth, or tongue)
def detect_smile(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    
    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        
        # Mouth landmarks
        top_lip = landmarks[13].y
        bottom_lip = landmarks[14].y
        left_corner = landmarks[0].x
        right_corner = landmarks[16].x
        
        # Detect mouth openness (a simple measure of smile)
        mouth_open = abs(top_lip - bottom_lip)
        
        # Check for smile based on mouth openness
        if mouth_open > 0.02:
            return True, "Smile detected"
        
        # Detect teeth exposure by checking mouth area brightness (we use a simple threshold)
        mouth_region = frame[int(landmarks[13].y * frame.shape[0]):int(landmarks[14].y * frame.shape[0]),
                             int(landmarks[0].x * frame.shape[1]):int(landmarks[16].x * frame.shape[1])]
        if np.mean(mouth_region) > 150:  # Assuming teeth are brighter than the rest of the mouth
            return True, "Teeth detected"
        
        # Detect tongue out by checking mouth width change or landmarks in the mouth
        tongue_visible = landmarks[13].y < landmarks[14].y and mouth_open > 0.03
        if tongue_visible:
            return True, "Tongue detected"
        
    return False, "No smile or feature detected"

# Helper function: encode to display
def convert_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    base64_img = base64.b64encode(buffer).decode()
    return f'data:image/jpeg;base64,{base64_img}'

# Main app function
def app():
    # Streamlit camera input
    camera_input = st.camera_input("Take a picture")

    if camera_input:
        # Convert the image received from Streamlit into an OpenCV format
        image = Image.open(camera_input)
        frame = np.array(image)

        # Detect smile, teeth, or tongue
        detected_feature, feature_message = detect_smile(frame)

        if detected_feature:
            # Save image to local folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_folder}/smile_{timestamp}.jpg"
            cv2.imwrite(filename, frame)

            st.balloons()
            st.image(frame, caption=f"{feature_message} 😁")
            st.success("Your smile lights up the world!")
            st.info("Have a wonderful day ✨")
        else:
            st.image(frame, caption="No smile or feature detected.")

# Run the app
if __name__ == "__main__":
    app()
