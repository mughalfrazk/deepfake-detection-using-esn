import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))

import cv2 as cv
import numpy as np
import mediapipe as mp

class FaceDetectionGenerator:
    def __init__(self, model_selection=1, min_detection_confidence=0.5):
        try:
            self.results = None
            self.model_selection = model_selection
            self.min_detection_confidence = min_detection_confidence

            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=self.model_selection,
                min_detection_confidence=min_detection_confidence,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to initialize FaceMeshGenerator: {str(e)}")

    def detect_and_crop_face(self, frame, skip_multiple_faces=False, logs=False):
        if frame is None:
            raise ValueError("Input frame cannot be None")

        height, width, _ = frame.shape

        try:
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.results = self.face_detection.process(frame_rgb)

            if self.results.detections is None:
                if logs is True:
                    print("No face detected. Skipping frame...")
                return None
            
            if len(self.results.detections) > 1 and skip_multiple_faces:
                if logs is True:
                    print("Multiple faces detected. Skipping frame...")
                return None

            im_bbox = self.results.detections[0].location_data.relative_bounding_box
            np_annotated_image = np.array(frame)
            xleft = im_bbox.xmin * width
            xtop = im_bbox.ymin * height
            xright = im_bbox.width * width + xleft
            xbottom = im_bbox.height * height + xtop

            xleft, xtop, xright, xbottom = (
                int(xleft),
                int(xtop),
                int(xright),
                int(xbottom),
            )

            return np_annotated_image[xtop:xbottom, xleft:xright]
        except Exception as e:
            raise RuntimeError(f"Error processing frame: {str(e)}")


def detect_face(video_path, resizing_factor):
    try:
        cap = cv.VideoCapture(0 if video_path == 0 else video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video capture")

        # Get video properties
        f_w, f_h, fps = (
            int(cap.get(x))
            for x in (
                cv.CAP_PROP_FRAME_WIDTH,
                cv.CAP_PROP_FRAME_HEIGHT,
                cv.CAP_PROP_FPS,
            )
        )

        detector_generator = FaceDetectionGenerator()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = detector_generator.detect_and_crop_face(frame)

            if video_path == 0:
                frame = cv.flip(frame, 1)

            if resizing_factor <= 0:
                raise ValueError("Resizing factor must be positive")

            # frame = cv.resize(
            #     frame, (int(f_w * resizing_factor), int(f_h * resizing_factor))
            # )
            frame = cv.resize(frame, (500, 500))
            cv.imshow("Video", frame)

            if cv.waitKey(1) & 0xFF == ord("p"):
                break

    except Exception as e:
        print(f"Error during video processing: {str(e)}")

    finally:
        if cap is not None:
            cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    video_path = "./sample_video.mp4"
    resizing_factor = 1 if video_path == 0 else 0.5
    detect_face(video_path, resizing_factor)
