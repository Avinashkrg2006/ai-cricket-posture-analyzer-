import mediapipe as mp
import cv2

class PoseDetector:
    def __init__(self, detection_conf=0.7, tracking_conf=0.7):
        self.mp_pose = mp.solutions.pose
        self.pose_model = self.mp_pose.Pose(
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.drawer = mp.solutions.drawing_utils

    def detect_pose(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose_model.process(frame_rgb)

    def draw_landmarks(self, frame, results):
        if results.pose_landmarks:
            self.drawer.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.drawer.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                self.drawer.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
            )
        return frame