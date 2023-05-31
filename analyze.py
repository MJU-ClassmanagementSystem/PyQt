import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import cv2
import dlib
import numpy as np
import tensorflow as tf
from keras.models import load_model
import mysql.connector as mc
from datetime import datetime


from student import Student

def connect_to_mysql():
    mydb = mc.connect(
        host="34.22.65.53",
        user="root",
        password="cms",
        database="cms"
    )
    return mydb



class MyApp(QWidget):
    def __init__(self):
        super().__init__()

        # 카메라 객체를 설정합니다
        self.cap = cv2.VideoCapture(0)

        # Load models and initialize variables
        self.init_models_and_vars()

        # 이미지를 표시할 라벨을 설정합니다
        self.label = QLabel(self)
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)

        # 타이머를 설정합니다
        self.timer = QTimer()
        self.timer.timeout.connect(self.viewCam)
        self.timer.start(10)

        self.setLayout(vbox)
        self.setWindowTitle("Face Recognition and Emotion Analysis with PyQt")

    def init_models_and_vars(self):
        # 얼굴 인식 모델 로드
        self.face_recognition_model = dlib.face_recognition_model_v1(
            "dlib_face_recognition_resnet_model_v1.dat"
        )
        # 얼굴 랜드마크 모델 로드
        self.landmark_detector = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat"
        )

        self.blink_counts = []

        # 얼굴 인식을 위한 Dlib face detector 생성
        self.face_detector = dlib.get_frontal_face_detector()

        # Load known face embeddings from .npy files in 'faces' directory
        import os
        import numpy as np
        self.known_embeddings = []
        self.known_labels = []
        self.students = {}
        faces_dir = "faces"
        for filename in os.listdir(faces_dir):
            if filename.endswith(".npy"):
                label = filename.rstrip(".npy")
                embedding = np.load(os.path.join(faces_dir, filename))
                self.known_embeddings.append(embedding)
                self.known_labels.append(label)
                self.blink_counts.append(0)
                self.students[label] = Student(label)

        # 감정 분석을 위한 모델 로드
        self.emotion_model = load_model("emotion_model.hdf5")
        self.emotion_labels = [
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Sad",
            "Surprise",
            "Neutral",
        ]


    def add_known_face_embedding(self, embedding, label):
        self.known_embeddings.append(embedding)
        self.known_labels.append(label)

    def calculate_eye_aspect_ratio(self, eye_landmarks):
        eye_landmarks = np.array([[p.x, p.y] for p in eye_landmarks])

        horizontal_dist1 = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        horizontal_dist2 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[2])
        horizontal_length = (horizontal_dist1 + horizontal_dist2) / 2

        vertical_dist1 = np.linalg.norm(eye_landmarks[5] - eye_landmarks[1])
        vertical_dist2 = np.linalg.norm(eye_landmarks[4] - eye_landmarks[2])
        vertical_length = (vertical_dist1 + vertical_dist2) / 2

        aspect_ratio = vertical_length / horizontal_length

        return aspect_ratio

    def viewCam(self):
        # OpenCV를 이용하여 카메라로부터 이미지를 캡쳐합니다
        ret, frame = self.cap.read()
        if ret:
          dlib_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

          # 얼굴 인식 수행
          faces = self.face_detector(dlib_frame)

          # 각 얼굴에 대해 얼굴 식별 및 감정 분석 수행
          for face in faces:
              # 얼굴 랜드마크 추출
              landmarks = self.landmark_detector(dlib_frame, face)
              # 얼굴 임베딩 추출
              embedding = self.face_recognition_model.compute_face_descriptor(
                  dlib_frame, landmarks
              )

              # 등록된 얼굴 임베딩과 비교하여 누구인지 판별
              distances = []
              for known_embedding in self.known_embeddings:
                  distance = np.linalg.norm(
                      np.array(embedding) - np.array(known_embedding)
                  )
                  distances.append(distance)

              # 판별 결과 출력
              min_distance_idx = np.argmin(distances)
              min_distance = distances[min_distance_idx]

              if min_distance <= 0.4:
                  label = self.known_labels[min_distance_idx]
              else:
                  label = "Unknown"
                  return

              # 얼굴 영역에 라벨 표시
              left, top, right, bottom = (
                  face.left(),
                  face.top(),
                  face.right(),
                  face.bottom(),
              )
              cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
              cv2.putText(
                  frame,
                  f"{label} {min_distance}",
                  (left, top - 10),
                  cv2.FONT_HERSHEY_SIMPLEX,
                  0.9,
                  (0, 255, 0),
                  2,
              )

              # 얼굴 영역 추출
              face_img = frame[top:bottom, left:right]

              # 감정 분석을 위해 이미지를 흑백으로 변환
              if face_img.size > 0:

                  gray_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

                  # 이미지를 모델에 전달하여 감정 예측
                  resized_img = cv2.resize(
                      gray_face_img, (64, 64), interpolation=cv2.INTER_AREA
                  )
                  img_array = tf.keras.preprocessing.image.img_to_array(resized_img)
                  img_array = np.expand_dims(img_array, axis=0)
                  img_array /= 255
                  predictions = self.emotion_model.predict(img_array)
                  max_index = np.argmax(predictions[0])
                  emotion = self.emotion_labels[max_index]
                  emotions = {k: v for k, v in zip(self.emotion_labels, predictions[0])}
                  self.students[label].update_emotions(emotions)
                  self.students[label].display_info()
                  # 감정 결과 출력
                  cv2.putText(
                      frame,
                      emotion,
                      (left, top - 40),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.9,
                      (0, 0, 255),
                      2,
                  )
                  i = 25
                  for e, v in emotions.items():
                      v = round(v, 2)
                      cv2.putText(
                          frame,
                          f"{e} {v}",
                          (left, bottom + i),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.5,
                          (0, 0, 255),
                          1,
                      )
                      i += 25
                  left_eye_landmarks = landmarks.parts()[36:42]
                  right_eye_landmarks = landmarks.parts()[42:48]

                  left_eye_aspect_ratio = self.calculate_eye_aspect_ratio(left_eye_landmarks)
                  right_eye_aspect_ratio = self.calculate_eye_aspect_ratio(
                      right_eye_landmarks
                  )
                  eye_aspect_ratio = (left_eye_aspect_ratio + right_eye_aspect_ratio) / 2

                  if eye_aspect_ratio < 0.25:
                      self.blink_counts[min_distance_idx] += 1
                      self.students[label].add_blink()

                  # 눈 깜빡임 횟수 출력
                  cv2.putText(
                      frame,
                      f"Blink count: {self.blink_counts[min_distance_idx]}",
                      (left, bottom),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      0.9,
                      (0, 255, 0),
                      2,
                  )

              # OpenCV 형식의 이미지를 Pixmap으로 변환합니다
              image = QImage(
                  frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888
              ).rgbSwapped()
              pixmap = QPixmap.fromImage(image)

              # Pixmap을 라벨에 표시합니다
              self.label.setPixmap(pixmap)

    def closeEvent(self, event):
      
        mydb = connect_to_mysql()   
        mycursor = mydb.cursor()

        # id와 password로 인증 확인
        current_time = datetime.now()
        query = "INSERT INTO emotion (angry, disgust, fear, happy, neutral, sad, surprise, student_id,date) VALUES (%s,%s, %s, %s, %s, %s, %s, %s, %s)"
        blink_query = "INSERT INTO focus (date, focus_rate, student_id) VALUES (%s,%s, %s)"
        
        
        for key, value in self.students.items():
            values = list(value.getStudentEmotion().values())
            values.append(key)
            values.append(current_time)
            mycursor.execute(query, tuple(values))
            if value.get_blink() != 0:
                blinks = [current_time, self.calculate_score(value.get_blink()), key]
                mycursor.execute(blink_query, tuple(blinks))
            
          # print(f"blink: {value.get_blink()}")
          # for k, v in value.getStudentEmotion().items():
          #   print(k, v)
            
        mydb.commit()
        # 연결 종료
        mycursor.close()
        mydb.close()
            
        event.accept()
        self.cap.release()
    
    def calculate_score(self,blink):
      if blink <= 100:
          score = 100
      elif blink <= 200:
          score = 100 - ((blink - 100) * 0.5)
      elif blink <= 300:
          score = 50 - ((blink - 200) * 0.5)   
      else: 
          score = 0
      return score


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
