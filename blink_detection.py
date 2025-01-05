import numpy as np
import cv2 as cv
from models.FaceLandmarkModule import FaceLandmarkGenerator
from utils.drawing import DrawingUtils
import os

class BlinkDetection:
  def __init__(self, video_path, ear_threshold, consec_frames, save_video=False, output_filename=None):
    self.generator = FaceLandmarkGenerator()
    self.video_path = video_path
    self.save_video = save_video
    self.output_filename = output_filename

    # Eyes landmark indices by mediapipe
    self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    # Selected landmarks for EAR (Eye Aspect Ratio) Calculation
    self.RIGHT_EYE_EAR = [33, 159, 158, 133, 153, 145]
    self.LEFT_EYE_EAR = [362, 380, 374, 263, 386, 385]

    self.ear_threshold = ear_threshold
    self.consec_frames = consec_frames
    self.blink_counter = 0
    self.frame_counter = 0

    self.GREEN_COLOR = (86, 241, 13)
    self.RED_COLOR = (30, 46, 209)

    if self.save_video and self.output_filename:
      save_dir = "data/videos/outputs"
      os.makedirs(save_dir, exist_ok=True)
      self.output_filename = os.path.join(save_dir, self.output_filename)

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
  
  def eye_aspect_ratio(self, eye_landmarks, landmarks):
    A = np.linalg.norm(np.array(landmarks[eye_landmarks[1]]) - np.array(landmarks[eye_landmarks[5]]))
    B = np.linalg.norm(np.array(landmarks[eye_landmarks[2]]) - np.array(landmarks[eye_landmarks[4]]))
    C = np.linalg.norm(np.array(landmarks[eye_landmarks[0]]) - np.array(landmarks[eye_landmarks[3]]))
    return (A + B) / (2.0 * C)

  def set_colors(self, ear):
    return self.RED_COLOR if ear < self.ear_threshold else self.GREEN_COLOR
  
  def draw_eye_landmarks(self, frame, landmarks, eye_landmarks, color):
    for loc in eye_landmarks:
      cv.circle(frame, (landmarks[loc]), 2, color, cv.FILLED)

  def process_video(self):
    try:
      cap = cv.VideoCapture(self.video_path)
      if not cap.isOpened():
        print(f"Failed to open video: {self. video_path}")
        raise IOError("Error: couldn't open the video!")

      w, h, fps = (int(cap.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS))

      if self.save_video:
        self.out = cv.VideoWriter(self.output_filename, cv.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

      while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
          break
          
        frame, face_landmarks = self.generator.create_face_mesh(frame, draw=False)

        if len(face_landmarks) > 0:
          right_ear = self.eye_aspect_ratio(self.RIGHT_EYE_EAR, face_landmarks)
          left_ear = self.eye_aspect_ratio(self.LEFT_EYE_EAR, face_landmarks)
          ear = (right_ear + left_ear) / 2.0

          self.update_blink_count(ear)

          color = self.set_colors(ear)

          self.draw_eye_landmarks(frame, face_landmarks, self.RIGHT_EYE, color)
          self.draw_eye_landmarks(frame, face_landmarks, self.LEFT_EYE, color)
          DrawingUtils.draw_text_with_bg(frame, f"Blinks: {self.blink_counter}", (0, 60),
                font_scale=2, thickness=3,
                bg_color=color, text_color=(0, 0, 0))

          if self.save_video:
            self.out.write(frame)
          
          resized_frame = cv.resize(frame, (1280, 720))
          cv.imshow("Blink Counter", resized_frame)
        
        if cv.waitKey(int(1000/fps)) & 0xFF == ord('p'):
          break
      
      cap.release()
      if self.save_video:
        self.out.release()
      cv.destroyAllWindows()

    except Exception as e:
      print(f"An error occured: {e}")

if __name__ == "__main__":
  input_video_path = "./01_02__talking_against_wall__YVGY8LOK.mp4"

  blink_detection = BlinkDetection(
    video_path=input_video_path,
    ear_threshold=0.3,
    consec_frames=3,
    save_video=False
  )

  blink_detection.process_video()