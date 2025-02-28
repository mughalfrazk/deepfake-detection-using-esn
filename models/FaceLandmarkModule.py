import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]
sys.path.append(str(root_path))

import cv2 as cv
import mediapipe as mp

class FaceLandmarkGenerator:
    def __init__(
        self, mode=False, num_faces=2, min_detection_con=0.5, min_track_con=0.5
    ):
        try:
            self.results = None
            self.mode = mode
            self.num_faces = num_faces
            self.min_detection_con = min_detection_con
            self.min_track_con = min_track_con

            self.mp_faceDetector = mp.solutions.face_mesh
            self.face_mesh = self.mp_faceDetector.FaceMesh(
                static_image_mode=self.mode,
                max_num_faces=self.num_faces,
                min_detection_confidence=self.min_detection_con,
                min_tracking_confidence=self.min_track_con,
            )

            self.mp_draw = mp.solutions.drawing_utils
            self.drawSpecs = self.mp_draw.DrawingSpec(thickness=1, circle_radius=1)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FaceMeshGenerator: {str(e)}")

    def create_face_mesh(self, frame, draw=True):
        if frame is None:
            raise ValueError("Input frame cannot be None")

        try:
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.results = self.face_mesh.process(frame_rgb)
            landmarks_dict = {}

            if self.results.multi_face_landmarks:
                for face_lms in self.results.multi_face_landmarks:
                    if draw:
                        self.mp_draw.draw_landmarks(
                            frame,
                            face_lms,
                            self.mp_faceDetector.FACEMESH_CONTOURS,
                            self.drawSpecs,
                            self.drawSpecs,
                        )
                    ih, iw, _ = frame.shape
                    for ID, lm in enumerate(face_lms.landmark):
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        landmarks_dict[ID] = (x, y)

            return frame, landmarks_dict
        except Exception as e:
            raise RuntimeError(f"Error processing frame: {str(e)}")


def generate_face_mesh(video_path, resizing_factor):
    try:
        cap = cv.VideoCapture(0 if video_path == 0 else video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video capture")

        # Get video properties
        f_w, f_h, _ = (
            int(cap.get(x))
            for x in (
                cv.CAP_PROP_FRAME_WIDTH,
                cv.CAP_PROP_FRAME_HEIGHT,
                cv.CAP_PROP_FPS,
            )
        )

        mesh_generator = FaceLandmarkGenerator()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame, _ = mesh_generator.create_face_mesh(frame)

            if video_path == 0:
                frame = cv.flip(frame, 1)

            if resizing_factor <= 0:
                raise ValueError("Resizing factor must be positive")

            resized_frame = cv.resize(
                frame, (int(f_w * resizing_factor), int(f_h * resizing_factor))
            )
            cv.imshow("Video", resized_frame)

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
    generate_face_mesh(video_path, resizing_factor)
