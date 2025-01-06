import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))

import cv2 as cv
import numpy as np
from models.FaceLandmarkModule import FaceLandmarkGenerator
from utils.drawing import DrawingUtils

class DetectBlinking:
    def __init__(self, video_path, ear_threshold, consec_frames):
        self.generator = FaceLandmarkGenerator()
        self.video_path = video_path
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
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

    def process_video(self):
        try:
            cap = cv.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {self.video_path}")
                raise IOError("Error: couldn't open the video!")

            landmark_generator = FaceLandmarkGenerator()
            _, _, fps = (int(cap.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame, landmarks = landmark_generator.create_face_mesh(frame, draw=False)

                if len(landmarks) > 0:
                    right_ear = self.eye_aspect_ratio(self.RIGHT_EYE_EAR, landmarks)
                    left_ear = self.eye_aspect_ratio(self.LEFT_EYE_EAR, landmarks)
                    ear = (right_ear + left_ear) / 2.0

                    self.update_blink_count(ear)
                    color = self.set_colors(ear)

                    self.draw_eye_landmarks(frame, landmarks, self.RIGHT_EYE, color)
                    self.draw_eye_landmarks(frame, landmarks, self.LEFT_EYE, color)
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

        except Exception as e:
            print(f"An error occured: {e}")


if __name__ == "__main__":
    video_path = "./sample_video.mp4"

    detect_blinking = DetectBlinking(video_path=video_path, ear_threshold=0.3, consec_frames=3)
    detect_blinking.process_video()
