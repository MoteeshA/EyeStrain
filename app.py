import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
from collections import deque
import smtplib
from email.message import EmailMessage

# 🔥 SAFE MEDIAPIPE IMPORT (CLOUD FIX)
from mediapipe.solutions import face_mesh as mp_face_mesh

# ------------------------------
# STREAMLIT CONFIG
# ------------------------------
st.set_page_config(page_title="Eye Stress Detection", layout="wide")
st.title("👁️ Eye Stress Detection System")
st.caption("Multi-face eye stress detection with calibration, blink count, recommendations & email")

# ======================================================
# 📧 EMAIL INPUT
# ======================================================
st.markdown("## 📧 Session Report Email")
receiver_email = st.text_input(
    "Receiver Email ID",
    placeholder="example@gmail.com",
    help="CSV will be emailed when camera stops"
)

SENDER_EMAIL = "yourgmail@gmail.com"
APP_PASSWORD = "your_app_password_here"

# ------------------------------
# UI
# ------------------------------
run = st.checkbox("▶️ Start Camera")
col_video, col_side = st.columns([2.5, 1.3])
FRAME_WINDOW = col_video.image([])
logs_placeholder = col_side.empty()
col_side.markdown("## 📈 Live Graphs")
graphs_placeholder = col_side.empty()

# ------------------------------
# SESSION STATE
# ------------------------------
if "final_log" not in st.session_state:
    st.session_state.final_log = []

# ------------------------------
# MEDIAPIPE (UNCHANGED LOGIC)
# ------------------------------
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ------------------------------
# CAMERA & SETTINGS
# ------------------------------
cap = cv2.VideoCapture(0)

CALIBRATION_TIME = 3
EAR_CLOSE_RATIO = 0.80
MIN_BLINK_FRAMES = 2
BLINK_COOLDOWN = 0.3

face_state = {}
history = {}
start_time = time.time()

# ------------------------------
# MAIN LOOP
# ------------------------------
while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not accessible")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    results = face_mesh.process(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    logs_md = "## 📋 Live Face Logs\n"
    graph_frames = []

    if results.multi_face_landmarks:
        for face_id, face_landmarks in enumerate(results.multi_face_landmarks):

            if face_id not in face_state:
                face_state[face_id] = {
                    "ear_buffer": deque(maxlen=5),
                    "baseline_ears": [],
                    "baseline": None,
                    "first_seen": time.time(),
                    "blink_frames": 0,
                    "is_closed": False,
                    "blink_count": 0,
                    "last_blink_time": 0
                }
                history[face_id] = []

            state = face_state[face_id]

            left_eye = np.array([[int(face_landmarks.landmark[i].x * w),
                                  int(face_landmarks.landmark[i].y * h)] for i in LEFT_EYE])
            right_eye = np.array([[int(face_landmarks.landmark[i].x * w),
                                   int(face_landmarks.landmark[i].y * h)] for i in RIGHT_EYE])

            raw_ear = (eye_aspect_ratio(left_eye) +
                       eye_aspect_ratio(right_eye)) / 2

            state["ear_buffer"].append(raw_ear)
            ear_value = np.mean(state["ear_buffer"])

            # ------------------------------
            # CALIBRATION
            # ------------------------------
            if state["baseline"] is None:
                if time.time() - state["first_seen"] < CALIBRATION_TIME:
                    state["baseline_ears"].append(ear_value)
                    cv2.putText(frame, "Calibrating...",
                                (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 255), 2)
                    continue
                else:
                    state["baseline"] = np.mean(state["baseline_ears"])

            baseline = state["baseline"]

            # ------------------------------
            # BLINK DETECTION
            # ------------------------------
            now = time.time()
            eye_closed = ear_value < baseline * EAR_CLOSE_RATIO

            if eye_closed:
                state["blink_frames"] += 1
                state["is_closed"] = True
            else:
                if state["is_closed"] and state["blink_frames"] >= MIN_BLINK_FRAMES:
                    if now - state["last_blink_time"] > BLINK_COOLDOWN:
                        state["blink_count"] += 1
                        state["last_blink_time"] = now
                state["blink_frames"] = 0
                state["is_closed"] = False

            # ------------------------------
            # STRESS SCORE
            # ------------------------------
            elapsed = time.time() - start_time
            blink_rate = (state["blink_count"] / elapsed) * 60 if elapsed > 0 else 0

            ear_drop = np.clip((baseline - ear_value) / baseline, 0, 1)
            blink_norm = np.clip((12 - blink_rate) / 12, 0, 1)
            stress_score = int(np.clip(ear_drop * 60 + blink_norm * 40, 0, 100))

            if stress_score >= 70:
                color, label = (0, 0, 255), "HIGH"
                recommendation = "Take a 20s break, look away & hydrate"
            elif stress_score >= 40:
                color, label = (0, 165, 255), "MILD"
                recommendation = "Blink more & adjust screen distance"
            else:
                color, label = (0, 255, 0), "NORMAL"
                recommendation = "Maintain posture & blink naturally"

            # ------------------------------
            # DRAW
            # ------------------------------
            cv2.polylines(frame, [left_eye], True, color, 3)
            cv2.polylines(frame, [right_eye], True, color, 3)

            cv2.putText(
                frame,
                f"{label} | {stress_score}/100 | Blinks: {state['blink_count']}",
                (30, 80 + face_id * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )

            # ------------------------------
            # LOGGING
            # ------------------------------
            history[face_id].append({"Time": int(elapsed), "Stress": stress_score})
            history[face_id] = history[face_id][-60:]

            st.session_state.final_log.append({
                "Time": int(elapsed),
                "Face": face_id + 1,
                "EAR": round(ear_value, 3),
                "Baseline": round(baseline, 3),
                "Blinks": state["blink_count"],
                "Stress": stress_score,
                "Level": label,
                "Recommendation": recommendation
            })

            logs_md += f"""
### 👤 Face {face_id+1}
- 👀 Blinks: **{state['blink_count']}**
- 😵 Stress: **{label} ({stress_score}/100)**
- 💡 Recommendation: _{recommendation}_
---
"""

            graph_frames.append(
                pd.DataFrame(history[face_id]).set_index("Time")
            )

    logs_placeholder.markdown(logs_md)
    if graph_frames:
        graphs_placeholder.line_chart(pd.concat(graph_frames), height=180)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# ------------------------------
# EMAIL ON STOP
# ------------------------------
cap.release()

if st.session_state.final_log and receiver_email:
    df = pd.DataFrame(st.session_state.final_log)
    csv_data = df.to_csv(index=False).encode()

    msg = EmailMessage()
    msg["Subject"] = "Eye Stress Detection Session Report"
    msg["From"] = SENDER_EMAIL
    msg["To"] = receiver_email
    msg.set_content("Attached is your eye stress detection session report.")

    msg.add_attachment(csv_data, maintype="text", subtype="csv",
                       filename="eye_stress_report.csv")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        st.success("📧 Session report emailed successfully!")
    except Exception as e:
        st.error(f"❌ Email failed: {e}")
