"""
Complete Pose Detection System with Error Handling and Progress Bar
Integrates all poses with random selection and score tracking
Added: optional output video saving and Hands 90 / Hands Up poses
Updated: robust logging to avoid UnicodeEncodeError on Windows consoles
Updated: Level system (1=Easy, 2=Medium, 3=Hard) with level progression
Enhanced: Level results screens and final comprehensive dashboard
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import logging
import time
import random
from functools import wraps
from typing import Optional, Tuple, List, Dict
import traceback
import sys
import io

# ==========================================
# 1. Logging System Setup (robust for Windows)
# ==========================================
stdout_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pose_detection.log', encoding='utf-8'),
        logging.StreamHandler(stdout_stream)
    ]
)
logger = logging.getLogger(__name__)


# ==========================================
# 2. Custom Exceptions
# ==========================================
class CameraError(Exception):
    pass

class CSVDataError(Exception):
    pass

class MediaPipeError(Exception):
    pass

class PoseValidationError(Exception):
    pass


# ==========================================
# 3. Safe Camera Handler
# ==========================================
class SafeCamera:
    def __init__(self, camera_id: int = 0, max_retries: int = 3):
        self.camera_id = camera_id
        self.max_retries = max_retries
        self.cap = None

    def __enter__(self):
        for attempt in range(self.max_retries):
            try:
                self.cap = cv2.VideoCapture(self.camera_id)
                if not self.cap.isOpened():
                    raise CameraError(f"Failed to open camera {self.camera_id}")
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    raise CameraError("Camera not reading frames properly")
                logger.info(f"Camera {self.camera_id} opened successfully")
                return self.cap
            except CameraError as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries}: {e}")
                if self.cap:
                    self.cap.release()
                if attempt == self.max_retries - 1:
                    raise CameraError(
                        f"Could not open camera after {self.max_retries} attempts.\n"
                        "1. Camera is connected properly\n"
                        "2. No other app is using the camera\n"
                        "3. Camera permissions are granted"
                    )

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera closed successfully")


# ==========================================
# 4. Safe CSV Loader
# ==========================================
class SafeCSVLoader:
    @staticmethod
    def load_csv(csv_path: str, required_columns: List[str] = None) -> Optional[pd.DataFrame]:
        try:
            if not os.path.exists(csv_path):
                logger.warning(f"CSV file not found: {csv_path}")
                return None
            if os.path.getsize(csv_path) == 0:
                logger.warning(f"CSV file is empty: {csv_path}")
                return None
            df = pd.read_csv(csv_path)
            if df.empty:
                logger.warning(f"No data in {csv_path}")
                return None
            if required_columns:
                missing_cols = set(required_columns) - set(df.columns)
                if missing_cols:
                    logger.warning(f"Missing columns in {csv_path}: {missing_cols}")
                    return None
            logger.info(f"Loaded {csv_path} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV {csv_path}: {e}")
            return None


# ==========================================
# 5. Safe MediaPipe Handler
# ==========================================
class SafeMediaPipe:
    @staticmethod
    def initialize_pose(min_detection: float = 0.6, min_tracking: float = 0.6):
        try:
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                min_detection_confidence=min_detection,
                min_tracking_confidence=min_tracking
            )
            logger.info("Pose detector initialized")
            return pose, mp_pose
        except Exception as e:
            raise MediaPipeError(f"Failed to initialize Pose: {e}")

    @staticmethod
    def initialize_hands(min_detection: float = 0.6, min_tracking: float = 0.6):
        try:
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                min_detection_confidence=min_detection,
                min_tracking_confidence=min_tracking
            )
            logger.info("Hands detector initialized")
            return hands, mp_hands
        except Exception as e:
            raise MediaPipeError(f"Failed to initialize Hands: {e}")

    @staticmethod
    def safe_process(detector, image: np.ndarray):
        try:
            if image is None or image.size == 0:
                return None
            results = detector.process(image)
            return results
        except Exception as e:
            logger.error(f"Error processing: {e}")
            return None


# ==========================================
# 6. Progress Bar Renderer
# ==========================================
class ProgressBarRenderer:
    @staticmethod
    def draw_progress_bar(frame, score: int, total: int, x: int = 50, y: int = 150,
                        width: int = 400, height: int = 30):
        try:
            percentage = min(100, int((score / total) * 100)) if total > 0 else 0
            filled_width = int((percentage / 100) * width)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
            color = (0, 255, 0) if percentage >= 80 else (0, 165, 255) if percentage >= 50 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + filled_width, y + height), color, -1)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 255), 2)
            text = f"Score: {score}/{total} ({percentage}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except Exception as e:
            logger.error(f"Error drawing progress bar: {e}")


# ==========================================
# 7. Pose Definitions with Level System
# ==========================================
class PoseDefinitions:

    POSES = {
        # ============ LEVEL 1 — EASY ============
        "Hands Down": {
            "csv": "HandsDown.csv",
            "description": "Keep both hands down relaxed",
            "duration": 3,
            "level": 1
        },
        "Hands Up": {
            "csv": "hands_up.csv",
            "description": "Both hands up straight with open palms",
            "duration": 3,
            "level": 1
        },
        "T-Pose": {
            "csv": "T-pose.csv",
            "description": "Stand in T-Pose (arms straight out)",
            "duration": 3,
            "level": 1
        },
        # ============ LEVEL 2 — MEDIUM ============
        "Hands 90": {
            "csv": "Hands 90.csv",
            "description": "Hands at 90 degrees with open palms",
            "duration": 4,
            "level": 2
        },
        "Right Up Left Down": {
            "csv": "Right up_left down.csv",
            "description": "Right hand up (open), left hand down",
            "duration": 4,
            "level": 2
        },
        "Left Up Right Down": {
            "csv": "Left up_right down.csv",
            "description": "Left hand up (open), right hand down",
            "duration": 4,
            "level": 2
        },
        # ============ LEVEL 3 — HARD ============
        "Muscle Pose": {
            "csv": "Muscle_Pose.csv",
            "description": "Flex your right arm (fist closed)",
            "duration": 5,
            "level": 3
        },
        "Left Up Right on Shoulder": {
            "csv": "Left up_Right on shoulder.csv",
            "description": "Left hand up, right hand on shoulder",
            "duration": 5,
            "level": 3
        },
        "Right Up Left on Shoulder": {
            "csv": "Right up_left on shoulder.csv",
            "description": "Right hand up, left hand on shoulder",
            "duration": 5,
            "level": 3
        },
    }

    @staticmethod
    def get_poses_by_level(level: int) -> Dict:
        return {
            name: data for name, data in PoseDefinitions.POSES.items()
            if data["level"] == level
        }

    @staticmethod
    def angle_between(p1, p2, p3):
        try:
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return np.degrees(angle)
        except Exception as e:
            logger.error(f"Error calculating angle: {e}")
            return 0

    @staticmethod
    def is_hand_open(hand_landmarks):
        try:
            tips = [8, 12, 16, 20]
            mcps = [5, 9, 13, 17]
            open_count = sum(
                1 for tip, mcp in zip(tips, mcps)
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y
            )
            return open_count >= 3
        except Exception as e:
            logger.error(f"Error checking hand open: {e}")
            return False

    @staticmethod
    def is_hand_closed(hand_landmarks):
        try:
            tips = [8, 12, 16, 20]
            mcps = [5, 9, 13, 17]
            return all(
                hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mcp].y
                for tip, mcp in zip(tips, mcps)
            )
        except Exception as e:
            logger.error(f"Error checking hand closed: {e}")
            return False

    @staticmethod
    def assign_hand_to_side(wrist, r_sh, l_sh):
        try:
            w = np.array([wrist.x, wrist.y])
            right = np.array([r_sh.x, r_sh.y])
            left = np.array([l_sh.x, l_sh.y])
            return "right" if np.linalg.norm(w - right) < np.linalg.norm(w - left) else "left"
        except Exception as e:
            logger.error(f"Error assigning hand: {e}")
            return "right"


# ==========================================
# 8. Universal Pose Checker
# ==========================================
class UniversalPoseChecker:
    def __init__(self):
        self.pose_detector, self.mp_pose = SafeMediaPipe.initialize_pose()
        self.hand_detector, self.mp_hands = SafeMediaPipe.initialize_hands()
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose_defs = PoseDefinitions()

    def check_pose(self, frame, pose_name: str, pose_data: Dict) -> Tuple[bool, np.ndarray]:
        try:
            csv_path = pose_data.get("csv")
            df = SafeCSVLoader.load_csv(csv_path)
            if df is None:
                return False, frame

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            pose_results = SafeMediaPipe.safe_process(self.pose_detector, image_rgb)
            hand_results = SafeMediaPipe.safe_process(self.hand_detector, image_rgb)
            image_rgb.flags.writeable = True
            frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            if pose_results and pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS
                )
            if hand_results and hand_results.multi_hand_landmarks:
                for hand_lm in hand_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_lm, self.mp_hands.HAND_CONNECTIONS
                    )

            is_correct = self._validate_pose(pose_name, pose_results, hand_results, df)
            return is_correct, frame
        except Exception as e:
            logger.error(f"Error checking pose: {e}")
            return False, frame

    def _validate_pose(self, pose_name: str, pose_results, hand_results, df) -> bool:
        try:
            if not pose_results or not pose_results.pose_landmarks:
                return False
            lm = pose_results.pose_landmarks.landmark
            tolerance = 0.03

            if "T-Pose" in pose_name:
                return self._check_tpose(lm, df)
            elif "Muscle" in pose_name:
                return self._check_muscle_pose(lm, hand_results, df)
            elif "Right Up" in pose_name and "Left Down" in pose_name:
                return self._check_right_up_left_down(lm, hand_results, df, tolerance)
            elif "Left Up" in pose_name and "Right Down" in pose_name:
                return self._check_left_up_right_down(lm, hand_results, df, tolerance)
            elif "Left Up" in pose_name and "Right on Shoulder" in pose_name:
                return self._check_left_up_right_shoulder(lm, hand_results, df, tolerance)
            elif "Right Up" in pose_name and "Left on Shoulder" in pose_name:
                return self._check_right_up_left_shoulder(lm, hand_results, df, tolerance)
            elif "Hands Down" in pose_name:
                return self._check_hands_down(lm, hand_results)
            elif "Hands 90" in pose_name:
                return self._check_hands_90(lm, hand_results, df)
            elif "Hands Up" in pose_name:
                return self._check_hands_up(lm, hand_results, df)
            return False
        except Exception as e:
            logger.error(f"Error validating pose: {e}")
            return False

    # ============ LEVEL 1 CHECKS ============

    def _check_tpose(self, lm, df) -> bool:
        try:
            coords_current = np.array([
                lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y
            ])
            coords_db = df[[
                'r_sh_x','r_sh_y','r_el_x','r_el_y','r_wr_x','r_wr_y',
                'l_sh_x','l_sh_y','l_el_x','l_el_y','l_wr_x','l_wr_y'
            ]].values
            distances = np.linalg.norm(coords_db - coords_current, axis=1)
            return np.any(distances <= 0.1)
        except:
            return False

    def _check_hands_down(self, lm, hand_results) -> bool:
        try:
            r_sh = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_sh = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_wr = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            l_wr = lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
            r_el = lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            l_el = lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]

            r_elbow_angle = self.pose_defs.angle_between(
                (r_sh.x, r_sh.y), (r_el.x, r_el.y), (r_wr.x, r_wr.y)
            )
            l_elbow_angle = self.pose_defs.angle_between(
                (l_sh.x, l_sh.y), (l_el.x, l_el.y), (l_wr.x, l_wr.y)
            )
            return (r_wr.y > r_sh.y and
                    l_wr.y > l_sh.y and
                    r_elbow_angle > 150 and
                    l_elbow_angle > 150)
        except Exception as e:
            logger.error(f"Error in hands down: {e}")
            return False

    def _check_hands_up(self, lm, hand_results, df) -> bool:
        try:
            correct_data = df[df['posture_correct'] == 1] if 'posture_correct' in df.columns else df
            if len(correct_data) == 0:
                return False
            avg_elbow_angle = correct_data['elbow_angle'].mean() if 'elbow_angle' in correct_data.columns else 180.0

            r_sh = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            r_el = lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            r_wr = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            l_sh = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            l_el = lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            l_wr = lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value]

            r_angle = self.pose_defs.angle_between(
                (r_sh.x, r_sh.y), (r_el.x, r_el.y), (r_wr.x, r_wr.y)
            )
            l_angle = self.pose_defs.angle_between(
                (l_sh.x, l_sh.y), (l_el.x, l_el.y), (l_wr.x, l_wr.y)
            )

            both_up = r_wr.y < r_sh.y and l_wr.y < l_sh.y
            both_straight = abs(r_angle - avg_elbow_angle) < 25 and abs(l_angle - avg_elbow_angle) < 25
            return both_up and both_straight
        except Exception as e:
            logger.error(f"Error in hands up check: {e}")
            return False

    # ============ LEVEL 2 CHECKS ============

    def _check_hands_90(self, lm, hand_results, df) -> bool:
        try:
            correct_data = df[df['posture_correct'] == 1] if 'posture_correct' in df.columns else df
            if len(correct_data) == 0:
                return False
            avg_elbow_angle = correct_data['elbow_angle'].mean() if 'elbow_angle' in correct_data.columns else 90.0

            r_sh = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            r_el = lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            r_wr = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            l_sh = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            l_el = lm[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            l_wr = lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value]

            r_angle = self.pose_defs.angle_between(
                (r_sh.x, r_sh.y), (r_el.x, r_el.y), (r_wr.x, r_wr.y)
            )
            l_angle = self.pose_defs.angle_between(
                (l_sh.x, l_sh.y), (l_el.x, l_el.y), (l_wr.x, l_wr.y)
            )

            both_90 = abs(r_angle - avg_elbow_angle) < 20 and abs(l_angle - avg_elbow_angle) < 20
            both_up = r_wr.y < r_sh.y and l_wr.y < l_sh.y

            hand_open = False
            if hand_results and getattr(hand_results, 'multi_hand_landmarks', None):
                for hand in hand_results.multi_hand_landmarks:
                    if self.pose_defs.is_hand_open(hand):
                        hand_open = True
                        break

            return both_90 and both_up and hand_open
        except Exception as e:
            logger.error(f"Error in hands 90 check: {e}")
            return False

    def _check_right_up_left_down(self, lm, hand_results, df, tolerance) -> bool:
        try:
            r_sh = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_sh = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_wr = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            l_wr = lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value]

            right_hand_open = False
            if hand_results and hand_results.multi_hand_landmarks:
                for hand in hand_results.multi_hand_landmarks:
                    wrist = hand.landmark[0]
                    side = self.pose_defs.assign_hand_to_side(wrist, r_sh, l_sh)
                    if side == "right":
                        right_hand_open = self.pose_defs.is_hand_open(hand)

            for row in df.itertuples():
                cond_right = (
                    abs(r_wr.y - row.right_wrist_y) < tolerance and
                    right_hand_open
                )
                cond_left = abs(l_wr.y - row.left_wrist_y) < tolerance
                if cond_right and cond_left:
                    return True
            return False
        except:
            return False

    def _check_left_up_right_down(self, lm, hand_results, df, tolerance) -> bool:
        try:
            r_sh = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_sh = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_wr = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            l_wr = lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value]

            left_hand_open = False
            if hand_results and hand_results.multi_hand_landmarks:
                for hand in hand_results.multi_hand_landmarks:
                    wrist = hand.landmark[0]
                    side = self.pose_defs.assign_hand_to_side(wrist, r_sh, l_sh)
                    if side == "left":
                        left_hand_open = self.pose_defs.is_hand_open(hand)

            for row in df.itertuples():
                cond_left = (
                    abs(l_wr.y - row.left_wrist_y) < tolerance and
                    left_hand_open == row.left_hand_open
                )
                if cond_left:
                    return True
            return False
        except:
            return False

    # ============ LEVEL 3 CHECKS ============

    def _check_muscle_pose(self, lm, hand_results, df) -> bool:
        try:
            shoulder = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            elbow = lm[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
            wrist = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]

            elbow_angle = self.pose_defs.angle_between(
                (shoulder.x, shoulder.y),
                (elbow.x, elbow.y),
                (wrist.x, wrist.y)
            )

            hand_closed = False
            if hand_results and hand_results.multi_hand_landmarks:
                for hand in hand_results.multi_hand_landmarks:
                    if self.pose_defs.is_hand_closed(hand):
                        hand_closed = True

            return 30 < elbow_angle < 100 and hand_closed
        except:
            return False

    def _check_left_up_right_shoulder(self, lm, hand_results, df, tolerance) -> bool:
        try:
            r_sh = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_sh = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_wr = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            l_wr = lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value]

            right_hand_open = left_hand_open = False
            if hand_results and hand_results.multi_hand_landmarks:
                for hand in hand_results.multi_hand_landmarks:
                    wrist = hand.landmark[0]
                    side = self.pose_defs.assign_hand_to_side(wrist, r_sh, l_sh)
                    if side == "right":
                        right_hand_open = self.pose_defs.is_hand_open(hand)
                    else:
                        left_hand_open = self.pose_defs.is_hand_open(hand)

            for row in df.itertuples():
                cond_right = (
                    abs(r_wr.y - row.right_wrist_y) < tolerance and
                    abs(r_sh.y - row.right_shoulder_y) < tolerance and
                    right_hand_open == row.right_hand_open
                )
                cond_left = (
                    abs(l_wr.y - row.left_wrist_y) < tolerance and
                    abs(l_sh.y - row.left_shoulder_y) < tolerance and
                    left_hand_open == row.left_hand_open
                )
                if cond_right and cond_left:
                    return True
            return False
        except Exception as e:
            logger.error(f"Error in left up right shoulder: {e}")
            return False

    def _check_right_up_left_shoulder(self, lm, hand_results, df, tolerance) -> bool:
        try:
            r_sh = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            l_sh = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_wr = lm[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
            l_wr = lm[self.mp_pose.PoseLandmark.LEFT_WRIST.value]

            right_hand_open = left_hand_open = False
            if hand_results and hand_results.multi_hand_landmarks:
                for hand in hand_results.multi_hand_landmarks:
                    wrist = hand.landmark[0]
                    side = self.pose_defs.assign_hand_to_side(wrist, r_sh, l_sh)
                    if side == "right":
                        right_hand_open = self.pose_defs.is_hand_open(hand)
                    else:
                        left_hand_open = self.pose_defs.is_hand_open(hand)

            for row in df.itertuples():
                cond_right = (
                    abs(r_wr.y - row.right_wrist_y) < tolerance and
                    abs(r_sh.y - row.right_shoulder_y) < tolerance and
                    right_hand_open == row.right_hand_open
                )
                cond_left = (
                    abs(l_wr.y - row.left_wrist_y) < tolerance and
                    abs(l_sh.y - row.left_shoulder_y) < tolerance and
                    left_hand_open == row.left_hand_open
                )
                if cond_right and cond_left:
                    return True
            return False
        except Exception as e:
            logger.error(f"Error in right up left shoulder: {e}")
            return False


# ==========================================
# 9. Level UI Renderer
# ==========================================
class LevelUIRenderer:
    LEVEL_COLORS = {
        1: (0, 200, 0),      # Green  — Easy
        2: (0, 165, 255),    # Orange — Medium
        3: (0, 0, 255),      # Red    — Hard
    }
    LEVEL_LABELS = {
        1: "LEVEL 1 - EASY",
        2: "LEVEL 2 - MEDIUM",
        3: "LEVEL 3 - HARD",
    }

    @staticmethod
    def draw_level_badge(frame, current_level: int, x: int = 20, y: int = 260):
        try:
            color = LevelUIRenderer.LEVEL_COLORS.get(current_level, (255, 255, 255))
            label = LevelUIRenderer.LEVEL_LABELS.get(current_level, f"LEVEL {current_level}")
            cv2.rectangle(frame, (x, y), (x + 230, y + 32), color, -1)
            cv2.rectangle(frame, (x, y), (x + 230, y + 32), (255, 255, 255), 2)
            cv2.putText(frame, label, (x + 8, y + 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
        except Exception as e:
            logger.error(f"Error drawing level badge: {e}")


# ==========================================
# 10. Results Display System
# ==========================================
class ResultsDisplay:
    """Display level results and final dashboard"""
    
    @staticmethod
    def show_motivational_screen(cap, completed_level: int, video_writer=None):
        """
        Display motivational message between levels (3 seconds)
        
        Args:
            cap: Camera capture object
            completed_level: Level just completed (1 or 2)
            video_writer: Optional video writer
        """
        try:
            messages = {
                1: [
                    "Great start! Ready for more challenge?",
                    "Level 1 complete! Let's level up!",
                    "Awesome! Time to increase the difficulty!"
                ],
                2: [
                    "Outstanding! One more level to go!",
                    "You're on fire! Final level awaits!",
                    "Impressive! Ready for the ultimate challenge?"
                ]
            }
            
            msg = random.choice(messages.get(completed_level, ["Keep going!"]))
            next_level = completed_level + 1
            
            colors = {2: (0, 165, 255), 3: (0, 0, 255)}
            next_color = colors.get(next_level, (255, 255, 255))
            
            start_time = time.time()
            while time.time() - start_time < 3.0:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    time.sleep(0.01)
                    continue
                
                h, w = frame.shape[:2]
                
                # Semi-transparent overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (25, 25, 25), -1)
                frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                
                # Main message
                cv2.putText(frame, msg, (w//2 - 300, h//2 - 50),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 3)
                
                # Next level indicator
                next_text = f"Preparing Level {next_level}..."
                cv2.putText(frame, next_text, (w//2 - 180, h//2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, next_color, 2)
                
                # Animated dots
                dots = "." * (int((time.time() - start_time) * 2) % 4)
                cv2.putText(frame, dots, (w//2 + 150, h//2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, next_color, 2)
                
                cv2.imshow("Pose Detection Game", frame)
                
                if video_writer:
                    video_writer.write(frame)
                
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            logger.error(f"Error showing motivational screen: {e}")
    
    @staticmethod
    def show_level_results(cap, level: int, score: int, total: int, duration: float, 
                        concentration: float, video_writer=None):
        """
        Display level completion screen for 5 seconds
        
        Args:
            cap: Camera capture object
            level: Level number (1-3)
            score: Score achieved
            total: Total poses
            duration: Time taken in seconds
            concentration: Concentration percentage (0-100)
            video_writer: Optional video writer
        """
        try:
            percentage = int((score / total) * 100) if total > 0 else 0
            passed = score >= 2  # Pass threshold
            
            level_colors = {1: (0, 200, 0), 2: (0, 165, 255), 3: (0, 0, 255)}
            color = level_colors.get(level, (255, 255, 255))
            
            # Display for 5 seconds
            start_time = time.time()
            while time.time() - start_time < 10.0:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    time.sleep(0.01)
                    continue
                
                h, w = frame.shape[:2]
                
                # Dark overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (20, 20, 20), -1)
                frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
                
                # Title
                title = f"Level {level} {'COMPLETED!' if passed else 'NEEDS RETRY'}"
                title_color = (0, 255, 0) if passed else (0, 165, 255)
                cv2.putText(frame, title, (w//2 - 280, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, title_color, 3)
                
                # Level badge
                cv2.rectangle(frame, (w//2 - 150, 150), (w//2 + 150, 200), color, -1)
                cv2.rectangle(frame, (w//2 - 150, 150), (w//2 + 150, 200), (255, 255, 255), 3)
                level_text = LevelUIRenderer.LEVEL_LABELS.get(level, f"LEVEL {level}")
                cv2.putText(frame, level_text, (w//2 - 130, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                # Results box
                box_y = 250
                box_h = 280
                cv2.rectangle(frame, (w//2 - 250, box_y), (w//2 + 250, box_y + box_h), 
                            (40, 40, 40), -1)
                cv2.rectangle(frame, (w//2 - 250, box_y), (w//2 + 250, box_y + box_h), 
                            color, 3)
                
                # Score
                y_offset = box_y + 60
                cv2.putText(frame, f"Score: {score}/{total}", (w//2 - 200, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                
                # Percentage bar
                bar_w = 400
                bar_x = w//2 - bar_w//2
                y_offset += 50
                cv2.rectangle(frame, (bar_x, y_offset), (bar_x + bar_w, y_offset + 30), 
                            (50, 50, 50), -1)
                filled = int((percentage / 100) * bar_w)
                bar_color = (0, 255, 0) if percentage >= 67 else (0, 165, 255)
                cv2.rectangle(frame, (bar_x, y_offset), (bar_x + filled, y_offset + 30), 
                            bar_color, -1)
                cv2.rectangle(frame, (bar_x, y_offset), (bar_x + bar_w, y_offset + 30), 
                            (255, 255, 255), 2)
                cv2.putText(frame, f"{percentage}%", (w//2 - 30, y_offset + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Time
                y_offset += 70
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                time_text = f"Time: {minutes:02d}:{seconds:02d}"
                cv2.putText(frame, time_text, (w//2 - 120, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
                
                # Concentration
                y_offset += 50
                conc_text = f"Concentration: {concentration:.1f}%"
                conc_color = (0, 255, 0) if concentration >= 70 else (0, 165, 255) if concentration >= 50 else (0, 100, 255)
                cv2.putText(frame, conc_text, (w//2 - 180, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, conc_color, 2)
                
                # Status message
                y_offset += 60
                if passed:
                    msg = "Great Job! Moving to next level..."
                    msg_color = (0, 255, 0)
                else:
                    msg = "Keep trying! You'll get it!"
                    msg_color = (0, 165, 255)
                cv2.putText(frame, msg, (w//2 - 240, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, msg_color, 2)
                
                cv2.imshow("Pose Detection Game", frame)
                
                if video_writer:
                    video_writer.write(frame)
                
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            logger.error(f"Error showing level results: {e}")
    
    @staticmethod
    def show_final_dashboard(cap, level_stats: List[Dict], video_writer=None):
        """
        Display comprehensive final dashboard with all game statistics
        
        Args:
            cap: Camera capture object
            level_stats: List of dictionaries containing stats for each level
            video_writer: Optional video writer
        """
        try:
            # Calculate overall stats
            total_score = sum(s['score'] for s in level_stats)
            total_poses = sum(s['total'] for s in level_stats)
            total_time = sum(s['duration'] for s in level_stats)
            avg_concentration = np.mean([s['concentration'] for s in level_stats])
            
            # Concentration improvement
            if len(level_stats) >= 2:
                conc_improvement = level_stats[-1]['concentration'] - level_stats[0]['concentration']
            else:
                conc_improvement = 0
            
            overall_percentage = int((total_score / total_poses) * 100) if total_poses > 0 else 0
            
            # Determine achievement level
            if overall_percentage >= 90:
                achievement = "OUTSTANDING!"
                ach_color = (0, 255, 255)  # Yellow
                emoji = "⭐⭐⭐"
            elif overall_percentage >= 75:
                achievement = "EXCELLENT!"
                ach_color = (0, 255, 0)  # Green
                emoji = "⭐⭐"
            elif overall_percentage >= 60:
                achievement = "GOOD JOB!"
                ach_color = (0, 165, 255)  # Orange
                emoji = "⭐"
            else:
                achievement = "KEEP PRACTICING!"
                ach_color = (100, 100, 255)  # Light Red
                emoji = "💪"
            
            # Display dashboard for 8 seconds or until user presses key
            start_time = time.time()
            while time.time() - start_time < 8.0:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    time.sleep(0.01)
                    continue
                
                h, w = frame.shape[:2]
                
                # Dark overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (15, 15, 15), -1)
                frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)
                
                # Main title - smaller
                cv2.putText(frame, "GAME COMPLETE!", (w//2 - 180, 45),
                           cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
                
                # Achievement badge - smaller
                cv2.rectangle(frame, (w//2 - 180, 65), (w//2 + 180, 105), ach_color, -1)
                cv2.rectangle(frame, (w//2 - 180, 65), (w//2 + 180, 105), (255, 255, 255), 2)
                cv2.putText(frame, achievement, (w//2 - 160, 90),
                           cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
                
                # Main stats box - more compact
                main_box_y = 120
                cv2.rectangle(frame, (50, main_box_y), (w - 50, main_box_y + 120), 
                             (40, 40, 40), -1)
                cv2.rectangle(frame, (50, main_box_y), (w - 50, main_box_y + 120), 
                             (100, 100, 255), 2)
                
                # Overall score - smaller font
                y_pos = main_box_y + 35
                cv2.putText(frame, f"Total Score: {total_score}/{total_poses} ({overall_percentage}%)", 
                           (70, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Score bar - smaller
                y_pos += 12
                bar_w = w - 140
                bar_x = 70
                cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_w, y_pos + 20), 
                             (50, 50, 50), -1)
                filled = int((overall_percentage / 100) * bar_w)
                cv2.rectangle(frame, (bar_x, y_pos), (bar_x + filled, y_pos + 20), 
                             ach_color, -1)
                cv2.rectangle(frame, (bar_x, y_pos), (bar_x + bar_w, y_pos + 20), 
                             (255, 255, 255), 2)
                
                # Time and concentration - more compact
                y_pos += 40
                mins = int(total_time // 60)
                secs = int(total_time % 60)
                cv2.putText(frame, f"Time: {mins:02d}:{secs:02d}", (70, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
                cv2.putText(frame, f"Avg Focus: {avg_concentration:.1f}%", 
                           (w//2 + 20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                
                # Level-by-level breakdown - more compact
                breakdown_y = main_box_y + 140
                cv2.putText(frame, "Level Performance:", (70, breakdown_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                level_colors = {1: (0, 200, 0), 2: (0, 165, 255), 3: (0, 0, 255)}
                
                for i, stats in enumerate(level_stats):
                    y_pos = breakdown_y + 30 + (i * 50)  # Reduced spacing from 60 to 50
                    level = stats['level']
                    color = level_colors.get(level, (255, 255, 255))
                    
                    # Level box - smaller
                    cv2.rectangle(frame, (70, y_pos - 20), (w - 70, y_pos + 20), 
                                 (30, 30, 30), -1)
                    cv2.rectangle(frame, (70, y_pos - 20), (w - 70, y_pos + 20), 
                                 color, 2)
                    
                    # Level info - smaller font
                    level_text = f"L{level}: {stats['score']}/{stats['total']}"
                    cv2.putText(frame, level_text, (90, y_pos + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Mini progress bar - smaller
                    mini_bar_w = 180
                    mini_bar_x = w//2 - 40
                    perc = int((stats['score'] / stats['total']) * 100) if stats['total'] > 0 else 0
                    cv2.rectangle(frame, (mini_bar_x, y_pos - 8), 
                                 (mini_bar_x + mini_bar_w, y_pos + 8), (50, 50, 50), -1)
                    mini_filled = int((perc / 100) * mini_bar_w)
                    cv2.rectangle(frame, (mini_bar_x, y_pos - 8), 
                                 (mini_bar_x + mini_filled, y_pos + 8), color, -1)
                    cv2.rectangle(frame, (mini_bar_x, y_pos - 8), 
                                 (mini_bar_x + mini_bar_w, y_pos + 8), (200, 200, 200), 1)
                    
                    # Concentration - smaller font
                    conc_text = f"Focus: {stats['concentration']:.0f}%"
                    cv2.putText(frame, conc_text, (w - 230, y_pos + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                
                # Focus improvement indicator - more compact
                improvement_y = breakdown_y + 30 + (len(level_stats) * 50) + 30
                if conc_improvement > 0:
                    improve_text = f"Focus Improved: +{conc_improvement:.1f}%! Keep it up!"
                    improve_color = (0, 255, 0)
                    arrow = "↑"
                elif conc_improvement < 0:
                    improve_text = f"Focus Variation: {conc_improvement:.1f}% - Stay consistent!"
                    improve_color = (0, 165, 255)
                    arrow = "↓"
                else:
                    improve_text = "Consistent Focus - Well done!"
                    improve_color = (255, 255, 0)
                    arrow = "→"
                
                cv2.putText(frame, f"{arrow} {improve_text}", (70, improvement_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, improve_color, 2)
                
                # Motivational message - smaller
                msg_y = improvement_y + 35
                if overall_percentage >= 80:
                    messages = [
                        "Amazing performance! You're a natural!",
                        "Fantastic work! Keep up the great focus!",
                        "Superb! Your concentration is top-notch!"
                    ]
                elif overall_percentage >= 60:
                    messages = [
                        "Great effort! Practice makes perfect!",
                        "Good job! You're improving with each level!",
                        "Well done! Keep pushing yourself!"
                    ]
                else:
                    messages = [
                        "Don't give up! Every attempt makes you better!",
                        "Keep practicing! You've got this!",
                        "Great start! Consistency is key!"
                    ]
                
                msg = random.choice(messages)
                cv2.putText(frame, msg, (w//2 - 280, msg_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)
                
                # Exit instruction - smaller
                cv2.putText(frame, "Press any key to exit...", (w//2 - 130, h - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                
                cv2.imshow("Pose Detection Game", frame)
                
                if video_writer:
                    video_writer.write(frame)
                
                if cv2.waitKey(33) != -1:  # Any key pressed
                    break
                    
        except Exception as e:
            logger.error(f"Error showing final dashboard: {e}")


# ==========================================
# 11. Main Pose Game System — Level-Based with Results
# ==========================================
def run_pose_game(save_path: Optional[str] = None):
    """
    Main pose game with 3 levels and comprehensive results display.
    """
    video_writer = None
    writer_size = None
    writer_path = None

    LEVEL_CONFIG = {
        1: {"poses_count": 3, "duration": 3, "pass_score": 2},
        2: {"poses_count": 3, "duration": 4, "pass_score": 2},
        3: {"poses_count": 3, "duration": 5, "pass_score": 2},
    }
    MAX_RETRIES_PER_LEVEL = 2

    try:
        print("=" * 60)
        print("       POSE DETECTION GAME  —  3 LEVELS")
        print("=" * 60)
        print("  Instructions:")
        print("    - Follow the pose shown on screen")
        print("    - Hold each pose for the required time")
        print("    - Score >= 2/3 on each level to advance")
        print("    - Press 'q' to quit at any time")
        print("-" * 60)
        print("  Level 1 (Easy)   — basic poses   | hold 3 s")
        print("  Level 2 (Medium) — two-hand poses | hold 4 s")
        print("  Level 3 (Hard)   — complex poses  | hold 5 s")
        print("=" * 60)

        checker = UniversalPoseChecker()
        progress_renderer = ProgressBarRenderer()
        level_ui = LevelUIRenderer()
        results_display = ResultsDisplay()

        available_all = {
            name: data for name, data in PoseDefinitions.POSES.items()
            if SafeCSVLoader.load_csv(data["csv"]) is not None
        }
        if not available_all:
            print("No pose CSV files found! Exiting.")
            return

        available_by_level = {1: {}, 2: {}, 3: {}}
        for name, data in available_all.items():
            available_by_level[data["level"]][name] = data

        for lvl in [1, 2, 3]:
            print(f"    Level {lvl}: {len(available_by_level[lvl])} poses available")

        print("\n  Starting in 3 seconds...\n")
        time.sleep(3)

        # Game state tracking
        current_level = 1
        level_retries = {1: 0, 2: 0, 3: 0}
        level_stats = []  # Store stats for each completed level
        game_quit = False

        with SafeCamera() as cap:
            # Test camera first
            ret, test_frame = cap.read()
            if not ret or test_frame is None:
                print("Error: Cannot read from camera!")
                return
            
            print(f"Camera opened successfully! Resolution: {test_frame.shape[1]}x{test_frame.shape[0]}")
            
            # Create normal window
            cv2.namedWindow("Pose Detection Game", cv2.WINDOW_NORMAL)
            
            while current_level <= 3 and not game_quit:
                cfg = LEVEL_CONFIG[current_level]
                poses_needed = cfg["poses_count"]
                hold_dur = cfg["duration"]
                pass_score = cfg["pass_score"]

                pool = available_by_level[current_level]
                if len(pool) == 0:
                    print(f"  [Level {current_level}] No poses available — skipping.")
                    current_level += 1
                    continue

                pose_sequence = random.sample(
                    list(pool.items()),
                    min(poses_needed, len(pool))
                )

                print(f"\n{'=' * 60}")
                print(f"  {LevelUIRenderer.LEVEL_LABELS[current_level]}")
                print(f"  Poses: {len(pose_sequence)} | Hold: {hold_dur}s | Need: {pass_score}/{len(pose_sequence)} to pass")
                if level_retries[current_level] > 0:
                    print(f"  (Retry {level_retries[current_level]}/{MAX_RETRIES_PER_LEVEL})")
                print(f"{'=' * 60}")
                time.sleep(2)

                # Level tracking
                level_score = 0
                current_pose_index = 0
                pose_start_time = time.time()
                pose_held_time = 0.0
                is_holding = False
                first_frame_obtained = False
                level_start_time = time.time()
                correct_frames = 0
                total_frames = 0

                # Pose loop
                while current_pose_index < len(pose_sequence) and not game_quit:
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        logger.warning("Invalid frame received, skipping...")
                        time.sleep(0.01)
                        continue
                    
                    # Validate frame dimensions
                    if len(frame.shape) != 3 or frame.shape[2] != 3:
                        logger.warning("Invalid frame shape, skipping...")
                        time.sleep(0.01)
                        continue

                    # Initialize VideoWriter
                    if save_path and not first_frame_obtained:
                        try:
                            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            if not fps or fps <= 0 or np.isnan(fps):
                                fps = 30.0
                            h, w = frame.shape[0], frame.shape[1]
                            writer_size = (w, h)
                            base, ext = os.path.splitext(save_path)
                            if ext.lower() not in [".mp4", ".avi", ".mov", ".mkv"]:
                                save_path = base + ".mp4"
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            video_writer = cv2.VideoWriter(save_path, fourcc, fps, writer_size)
                            writer_path = save_path
                            if not video_writer.isOpened():
                                logger.warning("VideoWriter failed to open.")
                                video_writer = None
                            else:
                                logger.info(f"Saving video to: {save_path}")
                        except Exception as e:
                            logger.error(f"VideoWriter init error: {e}")
                            video_writer = None
                        finally:
                            first_frame_obtained = True

                    pose_name, pose_data = pose_sequence[current_pose_index]
                    is_correct, frame = checker.check_pose(frame, pose_name, pose_data)

                    # Track concentration
                    total_frames += 1
                    if is_correct:
                        correct_frames += 1

                    current_time = time.time()
                    if is_correct:
                        if not is_holding:
                            is_holding = True
                            pose_start_time = current_time
                        pose_held_time = current_time - pose_start_time
                    else:
                        is_holding = False
                        pose_held_time = 0.0

                    if pose_held_time >= hold_dur:
                        level_score += 1
                        current_pose_index += 1
                        is_holding = False
                        pose_held_time = 0.0
                        print(f"    Pose {current_pose_index}/{len(pose_sequence)} done!")

                    # Draw UI
                    try:
                        lvl_color = LevelUIRenderer.LEVEL_COLORS.get(current_level, (255, 255, 255))
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (40, 40, 40), -1)
                        cv2.putText(frame, f"Pose: {pose_name}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(frame, pose_data["description"], (20, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                        status_text = "CORRECT - Hold it!" if is_correct else "INCORRECT"
                        status_color = (0, 255, 0) if is_correct else (0, 0, 255)
                        cv2.putText(frame, status_text, (20, 105),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                        progress_renderer.draw_progress_bar(
                            frame, level_score, len(pose_sequence), 50, 140, 400, 30
                        )

                        timer_pct = min(100, int((pose_held_time / hold_dur) * 100))
                        t_w = 400
                        t_filled = int((timer_pct / 100) * t_w)
                        cv2.rectangle(frame, (50, 185), (50 + t_w, 215), (50, 50, 50), -1)
                        cv2.rectangle(frame, (50, 185), (50 + t_filled, 215), (255, 165, 0), -1)
                        cv2.rectangle(frame, (50, 185), (50 + t_w, 215), (255, 255, 255), 2)
                        cv2.putText(frame, f"Hold: {pose_held_time:.1f}s / {hold_dur}s", (50, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        level_ui.draw_level_badge(frame, current_level, x=50, y=255)

                        cv2.putText(frame, f"Pose {current_pose_index + 1}/{len(pose_sequence)}",
                                    (frame.shape[1] - 210, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(frame, f"Level {current_level} / 3",
                                    (frame.shape[1] - 160, 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, lvl_color, 2)

                    except Exception as e:
                        logger.error(f"UI draw error: {e}")

                    cv2.imshow("Pose Detection Game", frame)

                    if video_writer is not None:
                        try:
                            if writer_size and (frame.shape[1], frame.shape[0]) != writer_size:
                                video_writer.write(cv2.resize(frame, writer_size))
                            else:
                                video_writer.write(frame)
                        except Exception as e:
                            logger.error(f"Video write error: {e}")
                            try:
                                video_writer.release()
                            except:
                                pass
                            video_writer = None

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        game_quit = True
                        break

                if game_quit:
                    break

                # Calculate level statistics
                level_duration = time.time() - level_start_time
                concentration = (correct_frames / total_frames * 100) if total_frames > 0 else 0
                
                # Store level stats
                level_stats.append({
                    'level': current_level,
                    'score': level_score,
                    'total': len(pose_sequence),
                    'duration': level_duration,
                    'concentration': concentration
                })

                print(f"\n  [Level {current_level}] Result: {level_score}/{len(pose_sequence)}", end="")

                # Show level results screen
                results_display.show_level_results(
                    cap, current_level, level_score, len(pose_sequence),
                    level_duration, concentration, video_writer
                )

                if level_score >= pass_score:
                    print("  --> PASSED!")
                    # Show motivational message before next level
                    if current_level < 3:  # Don't show before final dashboard
                        results_display.show_motivational_screen(
                            cap, current_level, video_writer
                        )
                    current_level += 1
                else:
                    if level_retries[current_level] < MAX_RETRIES_PER_LEVEL:
                        level_retries[current_level] += 1
                        print(f"  --> Retry {level_retries[current_level]}/{MAX_RETRIES_PER_LEVEL}")
                        time.sleep(2)
                    else:
                        print("  --> No retries left. Game Over.")
                        break

            # Show final dashboard
            if level_stats:
                results_display.show_final_dashboard(cap, level_stats, video_writer)

        # Final summary
        print("\n" + "=" * 60)
        if game_quit:
            print("  GAME STOPPED BY USER")
        elif current_level > 3:
            print("  YOU COMPLETED ALL 3 LEVELS!")
        else:
            print(f"  GAME OVER — stopped at Level {current_level}")
        print("=" * 60)
        
        if level_stats:
            total_score = sum(s['score'] for s in level_stats)
            total_poses = sum(s['total'] for s in level_stats)
            print(f"  Total Score : {total_score} / {total_poses}")
            if total_poses > 0:
                print(f"  Percentage  : {int((total_score / total_poses) * 100)}%")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\nGame stopped (KeyboardInterrupt)")
    except Exception as e:
        logger.error(f"Game error: {e}\n{traceback.format_exc()}")
        print(f"\nError: {e}")
    finally:
        if video_writer is not None:
            try:
                video_writer.release()
                logger.info(f"Video saved to: {writer_path}")
            except Exception as e:
                logger.error(f"Error releasing VideoWriter: {e}")
        cv2.destroyAllWindows()


# ==========================================
# 12. Main Entry Point
# ==========================================
if __name__ == "__main__":
    run_pose_game(save_path="output.mp4")