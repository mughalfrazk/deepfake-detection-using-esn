import numpy as np
import cv2 as cv
from models.FaceLandmarkModule import FaceLandmarkGenerator
import glob
import os

class BlinkDetectionAndEARPlot:
    # Facial landmark indices for eyes
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE_EAR = [33, 159, 158, 133, 153, 145]  # Points for EAR calculation
    LEFT_EYE_EAR = [362, 380, 374, 263, 386, 385]  # Points for EAR calculation

    def __init__(
        self,
        video_path,
        threshold,
        consec_frames
    ):
        self.generator = FaceLandmarkGenerator()
        self.video_path = video_path
        self.EAR_THRESHOLD = threshold
        self.CONSEC_FRAMES = consec_frames

        self.blink_counter = 0
        self.frame_counter = 0
        self.frame_number = 0
        self.ear_values = []
        self.frame_numbers = []

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

        return frame, ear

    def process_video(self):
        """Process the entire video and detect blinks."""
        try:
            cap = cv.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise IOError(f"Failed to open video: {self.video_path}")

            features_list = self._process_video_frames(cap)
            np_features_list = np.array(features_list)
            print(np_features_list.shape)

            return np_features_list

        except Exception as e:
            print(f"An error occurred: {e}")


    def _process_video_frames(self, cap):
        """Process individual frames from the video capture."""
        # Get video properties
        # w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        # h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        # frame_count = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        # fps = int(cap.get(cv.CAP_PROP_FPS))

        video_features = []
        previous_EAR = None
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame and get EAR
            frame, ear = self.process_frame(frame)

            if ear is not None:
                if previous_EAR is None:
                    delta_EAR = 0
                else:
                    delta_EAR = ear - previous_EAR

                previous_EAR = ear
                blink_indicator = int(ear < self.EAR_THRESHOLD)
                
                self._update_blink_detection(ear)
                print(f"Frame No.: {self.frame_number}, EAR: {ear}, Δ EAR: {delta_EAR}, Blink: {blink_indicator}")

                video_features.append([ear, delta_EAR, blink_indicator])

            if cv.waitKey(1) & 0xFF == ord("p"):
                break
        
        return video_features
    
    def _update_blink_detection(self, ear):
        """Update blink detection based on EAR value."""
        self.ear_values.append(ear)
        self.frame_numbers.append(self.frame_number)

        if ear < self.EAR_THRESHOLD:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.CONSEC_FRAMES:
                self.blink_counter += 1
            self.frame_counter = 0

        self.frame_number += 1

def get_features_and_save_npy(video_paths, output_dir):

    for idx, p in enumerate(video_paths):
        _, tail = os.path.split(p)
        name = tail.split(".")[0]

        output_filename = f"{name}.npy"
        path = output_dir
        if idx < 40:
            blink_counter = BlinkDetectionAndEARPlot(p, 0.294, 4)
            video_features = blink_counter.process_video()
            print("video_features: ", video_features.shape, len(video_features))

            if not(os.path.exists(f"{path}{output_filename}")):
                os.makedirs(path, exist_ok=True)
                ds = {"ORE_MAX_GIORNATA": 5}
                np.save(os.path.join(path, output_filename), ds)

            np.save(f"{path}/{output_filename}", video_features)
            # return video_features, f"{path}/{output_filename}"

if __name__ == "__main__":
    # Example usage
    fake_dir = "/Users/mughalfrazk/Study/SHU/Dissertation/code/mediapipe-eye-detection/dataset/manipulated_sequences"
    orig_dir = "/Users/mughalfrazk/Study/SHU/Dissertation/code/mediapipe-eye-detection/dataset/original_sequences"
    
    fake_paths = glob.glob(fake_dir + "/*/*/*/*.mp4")
    orig_paths = glob.glob(orig_dir + "/*/*/*/*.mp4")
    print(len(fake_dir))
    print(len(orig_dir))

    input_video_path = "/Users/mughalfrazk/Study/SHU/Dissertation/code/mediapipe-eye-detection/dataset/manipulated_sequences/DeepFakeDetection/c40/videos/28_08__secret_conversation__5DCAESDA.mp4"

    all_video_features = []
    all_video_targets = []

    get_features_and_save_npy(fake_paths, f"./out/")
    get_features_and_save_npy(orig_paths, f"./out/")
