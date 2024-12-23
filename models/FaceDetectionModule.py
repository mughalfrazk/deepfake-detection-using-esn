import cv2 as cv
import mediapipe as mp
import numpy as np


def crop_out_face(cv2_image_path, show_bbox=False):
    annotated_image = cv2_image_path.copy()
    height, width, _ = annotated_image.shape

    # Initialize Face Detection
    mp_face_detection = mp.solutions.face_detection

    # For static images:
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    # Convert the BGR image to RGB before processing.
    result = face_detection.process(cv.cvtColor(annotated_image, cv.COLOR_BGR2RGB))

    if not result.detections:
        return cv2_image_path

    if show_bbox:
        mp_drawing = mp.solutions.drawing_utils
        for detection in result.detections:
            mp_drawing.draw_detection(annotated_image, detection)

        return annotated_image

    im_bbox = result.detections[0].location_data.relative_bounding_box
    np_annotated_image = np.array(annotated_image)
    xleft = im_bbox.xmin * width
    xtop = im_bbox.ymin * height
    xright = im_bbox.width * width + xleft
    xbottom = im_bbox.height * height + xtop

    xleft, xtop, xright, xbottom = int(xleft), int(xtop), int(xright), int(xbottom)

    return np_annotated_image[xtop:xbottom, xleft:xright]


# v
# v
# v
# v


class FaceDetectionGenerator:
    def __init__(self, model_selection=0.5, min_detection_confidence=0.5):
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

    def crop_detected_face(self, frame):
        if frame is None:
            raise ValueError("Input frame cannot be None")

        height, width, _ = frame.shape

        try:
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.results = self.face_detection.process(frame_rgb)

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

            print(xtop, xbottom, xleft, xright)
            return np_annotated_image[xtop:xbottom, xleft:xright]
        except Exception as e:
            raise RuntimeError(f"Error processing frame: {str(e)}")


def detect_and_crop_video(video_path, resizing_factor, save_video=False, filename=None):
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

        # detector_generator = FaceDetectionGenerator()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # cropped_frame = detector_generator.crop_detected_face(frame)

            # if video_path == 0:
            #     frame = cv.flip(frame, 1)

            # if resizing_factor <= 0:
            #     raise ValueError("Resizing factor must be positive")

            # resized_frame = cv.resize(
            #     frame, (int(f_w * resizing_factor), int(f_h * resizing_factor))
            # )
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
    video_path = 0
    resizing_factor = 1 if video_path == 0 else 0.5
    detect_and_crop_video("/Users/mughalfrazk/Study/SHU/Dissertation/code/mediapipe-eye-detection/models/FaceLandmarkModule.py", resizing_factor)
