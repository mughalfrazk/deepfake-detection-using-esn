import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[2]
sys.path.append(str(root_path))

import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from models.FaceLandmarkModule import FaceLandmarkGenerator
from models.FaceDetectionModule import FaceDetectionGenerator
from utils.drawing import DrawingUtils
from sklearn.decomposition import PCA
class DetectBlinking:
    def __init__(self, video_path, ear_threshold, consec_frames, return_features=False, crop_face=False, process=True, logs=False, skip_multiple_faces=True):
        self.generator = FaceLandmarkGenerator()
        self.resnet = ResNet50(weights='./models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(100, 100, 3))
        self.video_path = video_path
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.return_features = return_features
        self.skip_multiple_faces = skip_multiple_faces
        self.crop_face = crop_face
        self.logs = logs
        self.process = process
        self.blink_counter = 0
        self.frame_counter = 0
        if self.crop_face:
            self.detection = FaceDetectionGenerator()

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
            cv.circle(frame, (landmarks[loc]), 1, color)

    def extract_features_with_resnet(self, frame):
        feature_map = self.resnet.predict(frame)
        flattened_features = feature_map.reshape(feature_map.shape[0], -1)
        return flattened_features

    def preprocess_frame(self, frame):
        # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # frame = frame.flatten()
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)
        return frame

    def process_frame(self, frame):
        """
        Process a single frame to detect and analyze eyes.

        Returns:
            tuple: Processed frame and EAR value
        """
        try:
            if self.crop_face: 
                frame = self.detection.detect_and_crop_face(frame, skip_multiple_faces=self.skip_multiple_faces, logs=self.logs)

                if frame is None or len(frame) == 0:
                    return None, None, None

                frame = cv.resize(frame, (100, 100), interpolation = cv.INTER_AREA)

            frame, face_landmarks = self.generator.create_face_mesh(frame, draw=False)

            if not face_landmarks:
                return None, None, None

            # Calculate EAR
            right_ear = self.eye_aspect_ratio(self.RIGHT_EYE_EAR, face_landmarks)
            left_ear = self.eye_aspect_ratio(self.LEFT_EYE_EAR, face_landmarks)
            ear = (right_ear + left_ear) / 2.0

            return frame, ear, face_landmarks
        except Exception as e:
            print(f"Exception occured in process_frame: {e}")

    def calculate_features(self, frame, ear, previous_EAR, video_features, ear_features):

        if ear is not None:
            if previous_EAR is None:
                delta_EAR = 0
            else:
                delta_EAR = ear - previous_EAR

            previous_EAR = ear
            blink_indicator = int(ear < self.ear_threshold)

            if self.process:
                frame = self.preprocess_frame(frame)
                features = self.extract_features_with_resnet(frame)
                if self.logs:
                    print(f"Frame Shape: {features.shape}, EAR: {ear}, Δ EAR: {delta_EAR}, Blink: {blink_indicator}")

                video_features.append(features[0])
                ear_features.append([ear, delta_EAR, blink_indicator])
            else:
                if self.logs:
                    print(f"EAR: {ear}, Δ EAR: {delta_EAR}, Blink: {blink_indicator}")
                video_features.append([ear, delta_EAR, blink_indicator])

        return video_features, previous_EAR, ear_features

    def process_video(self):
        try:
            cap = cv.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Failed to open video: {self.video_path}")
                raise IOError("Error: couldn't open the video!")

            _, _, fps = (int(cap.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS))

            video_features = []
            ear_features = []
            previous_EAR = None
            count = 0
            while cap.isOpened():
                count += 1
                ret, frame = cap.read()
                if not ret:
                    break

                frame, ear, face_landmarks = self.process_frame(frame)

                if frame is None:
                    if len(video_features) < 300:
                        print("Frame count: ", count)
                        return [], []
                    else:
                        return video_features, ear_features


                if self.return_features:
                    if ear is not None:
                        video_features, prev, ear_features = self.calculate_features(frame, ear, previous_EAR, video_features, ear_features)
                        previous_EAR = prev

                    if self.logs:
                        print("Frame count: ", count)
                else:
                    self.update_blink_count(ear)
                    color = self.set_colors(ear)

                    self.draw_eye_landmarks(frame, face_landmarks, self.RIGHT_EYE, color)
                    self.draw_eye_landmarks(frame, face_landmarks, self.LEFT_EYE, color)
                    if self.crop_face is False:    
                        DrawingUtils.draw_text_with_bg(
                            frame,
                            f"Blinks: {self.blink_counter}",
                            (0, 60),
                            font_scale=2,
                            thickness=3,
                            bg_color=color,
                            text_color=(0, 0, 0),
                        )

                    if not self.crop_face:
                        frame = cv.resize(frame, (1280, 720))

                    cv.imshow("Blink Counter", frame)
                    if cv.waitKey(int(1000 / fps)) & 0xFF == ord("p"):
                        break

            cap.release()
            cv.destroyAllWindows()

            return video_features, ear_features

        except Exception as e:
            print(f"An error occured in method process_video(): {e}")

def apply_pca(features, n_components=50):
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)

    return reduced_features

if __name__ == "__main__":
    video_path = "./sample_video.mp4"

    detect_blinking = DetectBlinking(
        video_path=video_path,
        ear_threshold=0.3,
        consec_frames=3,
        crop_face=False,
        return_features=False,
        process=False,
        logs=False
    )
    video_features, ear_features = detect_blinking.process_video()
    video_features = np.array(video_features)
    ear_features = np.array(ear_features)

    if len(video_features) > 0:
        print("Video Features: ", video_features.shape)
        print("Ear Features: ", ear_features.shape)
        reduced_features = apply_pca(video_features, n_components=50)
        print("Video Features: ", reduced_features.shape)
        final_features = np.concatenate((reduced_features, ear_features), axis=1)
        print("Final Features: ", final_features.shape)
        print("Ear Features: ", ear_features[1])
        print("Final Features: ", final_features[1][47:53])
    else:
        print("Skipping...")

