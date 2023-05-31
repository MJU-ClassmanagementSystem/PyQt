import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import cv2
import dlib
import numpy as np
import os
import pymysql
from datetime import datetime
# MySQL 서버에 연결
conn = pymysql.connect(
    host='localhost',
    user='root',
    password='',
    database='cms',
    cursorclass=pymysql.cursors.DictCursor
)

# 커서 생성
cursor = conn.cursor()
def check_attendance(student_id, attend_type, teacher_id="12345"):

    current_date = datetime.now().strftime('%Y-%m-%d')
    sql = "INSERT INTO attendance (attend_type, date, student_id, teacher_id) VALUES (%s, %s, %s, %s)"
    cursor.execute(sql, (attend_type, current_date, student_id, teacher_id))

    # 변경사항을 커밋
    conn.commit()
# 0 - 출첵, 1 - 결석, 2 - 실종



class Attendance(QWidget):
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
        self.timer.start(20)

        self.setLayout(vbox)
        self.setWindowTitle("Attendance")

    def init_models_and_vars(self):
        # 얼굴 인식 모델 로드
        self.face_recognition_model = dlib.face_recognition_model_v1(
            "dlib_face_recognition_resnet_model_v1.dat"
        )
        # 얼굴 랜드마크 모델 로드
        self.landmark_detector = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat"
        )

        # 얼굴 인식을 위한 Dlib face detector 생성
        self.face_detector = dlib.get_frontal_face_detector()

        self.known_embeddings = []
        self.known_labels = []
        self.attended = []
        self.load_known_faces()

    def load_known_faces(self):
        npy_files = [file for file in os.listdir() if file.endswith(".npy")]
        for file in npy_files:
            embedding = np.load(file)
            label = file.replace(".npy", "")
            self.known_embeddings.append(embedding)
            self.known_labels.append(label)
            self.attended.append(False)

    def viewCam(self):
        # OpenCV를 이용하여 카메라로부터 이미지를 캡쳐합니다
        ret, frame = self.cap.read()
        if ret:
            dlib_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 얼굴 인식 수행
            faces = self.face_detector(dlib_frame)

            # 각 얼굴에 대해 얼굴 식별 및 감정 분석 수행
            for face in faces:
                landmarks = self.landmark_detector(dlib_frame, face)
                unknown_embedding = self.face_recognition_model.compute_face_descriptor(dlib_frame, landmarks)
                unknown_embedding = np.array(unknown_embedding)

                min_distance = 1.0
                min_distance_index = -1
                for i, known_embedding in enumerate(self.known_embeddings):
                    distance = np.linalg.norm(unknown_embedding - known_embedding)
                    if distance < min_distance:
                        min_distance = distance
                        min_distance_index = i

                if min_distance < 0.6:
                    # check_attendance(self.known_labels[min_distance_index], 0)
                    if not self.attended[min_distance_index]:
                        check_attendance(self.known_labels[min_distance_index], 0)
                        self.attended[min_distance_index] = True
                        QMessageBox.information(self, "알림", f"{self.known_labels[min_distance_index]} 출석체크 되었습니다.")
                    self.attended[min_distance_index] = True

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
                    f"{self.known_labels[min_distance_index]} {min_distance}",
                    (left, top - 10),
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
        reply = QMessageBox.question(
            self, 'Message', 'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            for i in range(len(self.attended)):
                if not self.attended[i]:
                    check_attendance(self.known_labels[i], 1)
            self.close()
            event.accept()
            self.cap.release()
        else:
            event.ignore()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
