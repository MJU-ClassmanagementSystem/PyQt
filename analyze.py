import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout,QPushButton,QHBoxLayout, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import cv2
import dlib
import numpy as np
import tensorflow as tf
from keras.models import load_model
import mysql.connector as mc
from datetime import datetime
import os
import numpy as np
from student import Student

CLASS_DURATION = 2

def connect_to_mysql():
    mydb = mc.connect(
        host="34.22.65.53",
        user="root",
        password="cms",
        database="cms"
    )
    return mydb



class Analyze(QWidget):
    def __init__(self,teacher_id):
        super().__init__()
        self.teacher_id = teacher_id

        # 카메라 객체를 초기화합니다.
        self.cap = None

        # Load models and initialize variables
        self.init_models_and_vars()

        # 이미지를 표시할 라벨을 설정합니다
        self.label = QLabel(self)
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        
        # 버튼들을 생성합니다
        self.button_subjects = ["국어", "수학", "사회", "과학", "영어", "쉬는시간"]
        self.buttons = [QPushButton(subject, self) for subject in self.button_subjects]

        # 버튼들을 가로로 정렬합니다
        hbox_buttons = QHBoxLayout()
        for button in self.buttons:
            # if button.text() == "국어":
            #     button.clicked.connect(self.start_timer)
            button.clicked.connect(self.start_timer)
            hbox_buttons.addWidget(button)

        # QVBoxLayout에 버튼들을 추가합니다
        vbox.addLayout(hbox_buttons)

        # 타이머를 설정합니다
        self.timer = QTimer()
        self.timer.timeout.connect(self.viewCam)

        self.setLayout(vbox)
        self.setWindowTitle("Face Recognition and Emotion Analysis with PyQt")
        
        self.resize(800,600)
        


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


    def start_timer(self):
        self.cap = cv2.VideoCapture(0)
        # 클릭한 버튼의 텍스트를 가져옵니다
        self.subject_name = self.sender().text()
    
        # 타이머를 시작합니다
        if self.subject_name == "쉬는시간":
            # 쉬는시간은 1분
            self.timer.start(20)
            # QTimer.singleShot(2 * 60 * 1000, self.stop_timer_and_store_data)
            QTimer.singleShot(1 * 60 * 1000, self.stop_timer_and_store_data)
            
            for button in self.buttons:
                button.setDisabled(True)
        
        else:
            # 수업시간은 2분
            self.timer.start(20)
            # QTimer.singleShot(2 * 60 * 1000, self.stop_timer_and_store_data)
            QTimer.singleShot(CLASS_DURATION * 60 * 1000, self.stop_timer_and_store_data)
            
            for button in self.buttons:
                button.setDisabled(True)


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

              # Pixmap을 조정하여 라벨에 표시합니다
              # scaled_pixmap = pixmap.scaled(640, 480, Qt.KeepAspectRatio)
              # self.label.setPixmap(scaled_pixmap)
              self.label.setPixmap(pixmap)

    def closeEvent(self, event):
        event.accept()
        # self.cap.release()
        
        
    def get_teacher_id(self):
    # 로그인 시 사용한 user_id를 가져오는 함수를 구현하세요
    # 예를 들어, 로그인할 때 저장한 user_id를 반환하도록 합니다
    # 이 예시에서는 임의로 "teacher123"을 반환하도록 설정하였습니다
        return self.teacher_id
    
    def calculate_concentration_score(self, count, minute):
        average_blink_per_minute = 15  # 1분에 평균 눈 깜빡임 수

        total_blink = count / minute  # 분당 평균 눈 깜빡임 수 계산
        concentration_score = max(0, ((average_blink_per_minute - total_blink) / average_blink_per_minute) * 50 + 50)
        return concentration_score
    
    def stop_timer_and_store_data(self):
        # 타이머를 중지합니다
        self.timer.stop()
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        # DB에 데이터를 저장합니다
        mydb = connect_to_mysql()
        mycursor = mydb.cursor()

        current_time = datetime.now()
        query = "INSERT INTO emotion (angry, disgust, fear, happy, neutral, sad, surprise, student_id, date, subject_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        blink_query = "INSERT INTO focus (date, focus_rate, student_id, subject_id) VALUES (%s, %s, %s, %s)"
        subjectid_query = "SELECT id from subject where teacher_id = %s and subject_name = %s"
        
        mycursor.execute(subjectid_query, (self.get_teacher_id(), self.subject_name))
        subjectid = mycursor.fetchone()
        

        
        if subjectid is not None:
            subjectid = subjectid[0]

        for key, value in self.students.items():
            if value.get_blink() != 0:
                emotion_data = list(value.getStudentEmotion().values())
                emotion_data.append(key)
                # emotion_data.append(current_time)
                emotion_data.append("2023-06-05")
                emotion_data.append(subjectid)
                mycursor.execute(query, tuple(emotion_data))
                # blink_data = [current_time, self.calculate_concentration_score(value.get_blink(),CLASS_DURATION), key]
                blink_data = ["2023-06-05", self.calculate_concentration_score(value.get_blink(),CLASS_DURATION), key]
                blink_data.append(subjectid)
                mycursor.execute(blink_query, tuple(blink_data))

        mydb.commit()
        mycursor.close()
        mydb.close()
        self.students = {}
        self.blink_counts = []
        for i in self.known_labels:
            self.students[i] = Student(i)
            self.blink_counts.append(0)
        # 수업 종료 메시지를 표시합니다
        msg = QMessageBox()
        msg.setWindowTitle("알림")
        msg.setText("수업이 종료되었습니다.")
        msg.exec_()
        
        # 라벨에 표시된 이미지를 비웁니다
        self.label.clear()
        
        for button in self.buttons:
            button.setDisabled(False)
        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Analyze('test')
    ex.show()
    sys.exit(app.exec_())
