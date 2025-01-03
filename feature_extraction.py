import numpy as np

import cv2 as cv
import mediapipe as mp
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
                return False
                raise IOError(f"Failed to open video: {self.video_path}")

            features_list = self._process_video_frames(cap)
            np_features_list = np.array(features_list)
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

            # frame = self._crop_out_face(frame.copy())
            # frame = cv.resize(frame, (128, 128))

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

                # cv.imshow("Blink Counter", frame)
                
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

    def _crop_out_face(self, annotated_image):
        try:
            height, width, _ = annotated_image.shape
            mp_face_detection = mp.solutions.face_detection
            face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            result = face_detection.process(cv.cvtColor(annotated_image, cv.COLOR_BGR2RGB))

            if not result.detections:
                return annotated_image

            im_bbox = result.detections[0].location_data.relative_bounding_box
            np_annotated_image = np.array(annotated_image)
            xleft = im_bbox.xmin * width
            xtop = im_bbox.ymin*height
            xright = im_bbox.width * width + xleft
            xbottom = im_bbox.height*height + xtop

            xleft, xtop, xright, xbottom = int(xleft), int(xtop), int(xright), int(xbottom)

            return np_annotated_image[xtop:xbottom, xleft:xright]
        except Exception as e:
            print(f"Error in cropping image: {e}")


def faces_count(path):
    try:
        cap = cv.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_image = frame.copy()

            # Initialize Face Detection
            mp_face_detection = mp.solutions.face_detection

            # For static images:
            face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

            # Convert the BGR image to RGB before processing.
            result = face_detection.process(cv.cvtColor(annotated_image, cv.COLOR_BGR2RGB))

            return len(result.detections)

    except Exception as e:
        print(f"An error occurred: {e}")

def get_and_save_features(p, path, output_filename, idx):
    detected_faces = faces_count(p)
    if detected_faces != 1: return
    else:
        blink_counter = BlinkDetectionAndEARPlot(p, 0.294, 4)
        video_features = blink_counter.process_video()
        print("--------------")
        print("--------------")
        print("--------------")
        print("--------------")
        print(f"{idx} Video Processed | Features: ", video_features.shape, len(video_features))
        print("--------------")
        print("--------------")
        print("--------------")
        print("--------------")

        if not(os.path.exists(f"{path}{output_filename}")):
            os.makedirs(path, exist_ok=True)
            ds = {"ORE_MAX_GIORNATA": 5}
            np.save(os.path.join(path, output_filename), ds)

        np.save(f"{path}/{output_filename}", video_features)

        # return video_features, f"{path}/{output_filename}"

def get_features_and_save_npy(video_paths, output_dir):
    for idx, p in enumerate(video_paths):
        _, tail = os.path.split(p)
        name = tail.split(".")[0]

        np_path = output_dir + f"{name}.npy"
        file_already_exist = os.path.isfile(np_path)
        output_filename = f"{name}.npy"
        path = output_dir
        
        # if idx < 20:
        if not file_already_exist:
            get_and_save_features(p, path, output_filename, idx)
        else:
            print(f"{idx} => File missed: ", np_path)


if __name__ == "__main__":
    # Example usage
    fake_dir = "dataset/manipulated_sequences"
    orig_dir = "dataset/original_sequences"
    
    fake_paths = glob.glob(fake_dir + "/*/*/*/*.mp4")
    orig_paths = glob.glob(orig_dir + "/*/*/*/*.mp4")
    print("fake_paths: ", len(fake_paths))
    print("orig_dir: ", len(orig_paths))

    input_video_path = "dataset/manipulated_sequences/DeepFakeDetection/c40/videos/28_08__secret_conversation__5DCAESDA.mp4"

    all_video_features = []
    all_video_targets = []

    get_features_and_save_npy(fake_paths, f"./out/0/")
    # get_features_and_save_npy(orig_paths, f"./out/1/")