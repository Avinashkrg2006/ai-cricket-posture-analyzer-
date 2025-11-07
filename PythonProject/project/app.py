import time

import cv2
import streamlit as st

from angle_utils import calculate_angle
from pose_module import PoseDetector
from voice_feedback import speak

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Cricket Posture Analyzer", layout="wide")
st.title("🏏 AI Cricket Posture Analyzer")
st.markdown("### Analyze your *Batting* and *Bowling* Posture in Real-Time using AI 🤖")

# Sidebar
st.sidebar.header("📊 Live Posture Feedback")
mode = st.sidebar.radio("Select Mode:", ["Batting", "Bowling"])
run = st.checkbox("▶ Start Analysis")

posture_status = st.sidebar.empty()
feedback_box = st.sidebar.empty()
angle_box = st.sidebar.empty()
st.sidebar.markdown("---")
accuracy_bar = st.sidebar.progress(0)
accuracy_text = st.sidebar.empty()

# ------------------ RULES ------------------
BAT_RULES = {"elbow": (90, 130), "knee": (80, 110)}
BOWL_RULES = {"shoulder": (100, 140), "spine": (40, 70)}

# ------------------ INITIALIZE ------------------
detector = PoseDetector()
FRAME_WINDOW = st.image([])

# ------------------ MAIN LOGIC ------------------
if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Camera not found!")
        st.stop()

    st.info("📸 Camera started — press Stop to end session.")
    prev_time, fps_limit = 0, 1/20  # ≈20 FPS
    correct_frames, total_frames = 0, 0

    while run and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("⚠ Unable to access camera.")
            break

        # FPS throttle
        now = time.time()
        if now - prev_time < fps_limit:
            continue
        prev_time = now

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 360))
        results = detector.detect_pose(frame)
        landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None

        feedback, status = "", ""
        elbow_angle = knee_angle = shoulder_angle = spine_angle = 0

        if landmarks:
            mp_pose = detector.mp_pose
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # ---- Angles ----
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            knee_angle = calculate_angle(hip, knee, ankle)
            shoulder_angle = calculate_angle(elbow, shoulder, hip)
            spine_angle = calculate_angle(shoulder, hip, knee)

            # ---- Posture Evaluation ----
            if mode == "Batting":
                elbow_ok = BAT_RULES["elbow"][0] <= elbow_angle <= BAT_RULES["elbow"][1]
                knee_ok = BAT_RULES["knee"][0] <= knee_angle <= BAT_RULES["knee"][1]
                if elbow_ok and knee_ok:
                    status, feedback = "✅ Correct Batting Posture", "Perfect stance!"
                else:
                    status = "❌ Incorrect Batting Posture"
                    feedback = ("Raise your back elbow." if not elbow_ok
                                else "Bend your front knee slightly less.")
            else:  # Bowling
                shoulder_ok = BOWL_RULES["shoulder"][0] <= shoulder_angle <= BOWL_RULES["shoulder"][1]
                spine_ok = BOWL_RULES["spine"][0] <= spine_angle <= BOWL_RULES["spine"][1]
                if shoulder_ok and spine_ok:
                    status, feedback = "✅ Correct Bowling Posture", "Good bowling action!"
                else:
                    status = "❌ Incorrect Bowling Posture"
                    feedback = ("Lower your bowling arm slightly." if not shoulder_ok
                                else "Keep your spine straighter.")

            # ---- Accuracy Tracking ----
            total_frames += 1
            if "✅" in status:
                correct_frames += 1
            accuracy = (correct_frames / total_frames) * 100 if total_frames else 0

            # ---- Sidebar Updates ----
            posture_status.success(status) if "✅" in status else posture_status.error(status)
            feedback_box.write(f"🗣 {feedback}")
            angle_box.markdown(
                f"*Elbow:* {int(elbow_angle)}° | *Knee:* {int(knee_angle)}° | "
                f"*Shoulder:* {int(shoulder_angle)}° | *Spine:* {int(spine_angle)}°"
            )
            accuracy_bar.progress(min(int(accuracy), 100))
            accuracy_text.markdown(f"*Form Accuracy:* {accuracy:.1f}%")

            # ---- Voice ----
            if "❌" in status:
                speak(feedback)

            # ---- Draw Pose ----
            frame = detector.draw_landmarks(frame, results)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        time.sleep(0.01)

    cap.release()
    st.success("🏁 Camera stopped.")

    if total_frames:
        final_acc = (correct_frames / total_frames) * 100
        st.success(f"🎯 Session Accuracy: {final_acc:.2f}%")
else:
    st.warning("Click *Start Analysis* to begin.")