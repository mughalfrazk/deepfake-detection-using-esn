import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))

import cv2 as cv
import numpy as np
from models.FaceLandmarkModule import FaceLandmarkGenerator
from utils.drawing import DrawingUtils

class DetectBlinking:
    def __init__(self, video_path, ear_threshold, consec_frames, return_features=False):
        self.generator = FaceLandmarkGenerator()
        self.video_path = video_path
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.return_features = return_features
        self.blink_counter = 0
        self.frame_counter = 0

        # Eyes landmark indices by mediapipe
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

        # Selected landmarks for EAR (Eye Aspect Ratio) Calculation
        self.RIGHT_EYE_EAR = [33, 159, 158, 133, 153, 145]
        self.LEFT_EYE_EAR = [362, 380, 374, 263, 386, 385]

        self.GREEN_COLOR = (86, 241, 13)
        self.RED_COLOR = (30, 46, 209)

    def eye_aspect_ratio(self, eye_landmarks, landmarks):
        A = np.linalg.norm(
            np.array(landmarks[eye_landmarks[1]])
            - np.array(landmarks[eye_landmarks[5]])
        )
        B = np.linalg.norm(
            np.array(landmarks[eye_landmarks[2]])
            - np.array(landmarks[eye_landmarks[4]])
        )
        C = np.linalg.norm(
            np.array(landmarks[eye_landmarks[0]])
            - np.array(landmarks[eye_landmarks[3]])
        )
        return (A + B) / (2.0 * C)

    def update_blink_count(self, ear):
        blink_detected = False

        if ear < self.ear_threshold:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.consec_frames:
                self.blink_counter += 1
                blink_detected = True
            self.frame_counter = 0

        return blink_detected

    def set_colors(self, ear):
        return self.RED_COLOR if ear < self.ear_threshold else self.GREEN_COLOR

    def draw_eye_landmarks(self, frame, landmarks, eye_landmarks, color):
        for loc in eye_landmarks:
            cv.circle(frame, (landmarks[loc]), 2, color, cv.FILLED)

    def process_frame(self, frame):
        """
        Process a single frame to detect and analyze eyes.

        Returns:
            tuple: Processed frame and EAR value
        """
        frame, face_landmarks = self.generator.create_face_mesh(frame, draw=False)

        if not face_landmarks:
            return frame, None

        # Calculate EAR
        right_ear = self.eye_aspect_ratio(self.RIGHT_EYE_EAR, face_landmarks)
        left_ear = self.eye_aspect_ratio(self.LEFT_EYE_EAR, face_landmarks)
        ear = (right_ear + left_ear) / 2.0

        return frame, ear, face_landmarks

    def calculate_features(self, ear, previous_EAR, video_features):

        if ear is not None:
            if previous_EAR is None:
                delta_EAR = 0
            else:
                delta_EAR = ear - previous_EAR

            previous_EAR = ear
            blink_indicator = int(ear < self.ear_threshold)
            print(f"EAR: {ear}, Δ EAR: {delta_EAR}, Blink: {blink_indicator}")
            video_features.append([ear, delta_EAR, blink_indicator])

        return video_features, previous_EAR


    def process_video(self):
        try:
            cap = cv.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {self.video_path}")
                raise IOError("Error: couldn't open the video!")

            _, _, fps = (int(cap.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS))

            video_features = []
            previous_EAR = None
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame, ear, face_landmarks = self.process_frame(frame)

                if self.return_features:
                    if ear is not None:
                        video_features, prev = self.calculate_features(ear, previous_EAR, video_features)
                        previous_EAR = prev
                else:
                    self.update_blink_count(ear)
                    color = self.set_colors(ear)

                    self.draw_eye_landmarks(frame, face_landmarks, self.RIGHT_EYE, color)
                    self.draw_eye_landmarks(frame, face_landmarks, self.LEFT_EYE, color)
                    DrawingUtils.draw_text_with_bg(
                        frame,
                        f"Blinks: {self.blink_counter}",
                        (0, 60),
                        font_scale=2,
                        thickness=3,
                        bg_color=color,
                        text_color=(0, 0, 0),
                    )

                    resized_frame = cv.resize(frame, (1280, 720))
                    cv.imshow("Blink Counter", resized_frame)
                    if cv.waitKey(int(1000 / fps)) & 0xFF == ord("p"):
                        break

            cap.release()
            cv.destroyAllWindows()

            return np.array(video_features)

        except Exception as e:
            print(f"An error occured: {e}")


if __name__ == "__main__":
    video_path = "./sample_video.mp4"

    detect_blinking = DetectBlinking(video_path=video_path, ear_threshold=0.3, consec_frames=3, return_features=True)
    video_features = detect_blinking.process_video()
    print("Video Features: ", len(video_features), video_features.shape)
