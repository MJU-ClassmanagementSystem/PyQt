import sys
import mysql.connector
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton, QMessageBox, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import cv2
import dlib
import numpy as np

db_config = {
    'host': '34.22.65.53',
    'user': 'root',
    'password': 'cms',
    'database': 'cms'
}

class Register(QWidget):
    def __init__(self, user_id):
        super().__init__()

        # 카메라 객체를 설정합니다
        self.cap = cv2.VideoCapture(0)

        # L oad models and initialize variables
        self.init_models_and_vars()

        # MySQL에 연결합니다
        self.db_connection = mysql.connector.connect(**db_config)

        # 이미지를 표시할 라벨을 설정합니다
        self.label = QLabel(self)
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)

        # 이름과 학번을 입력할 수 있는 LineEdit과 해당 정보를 표시할 QLabel을 추가합니다
        hbox1 = QHBoxLayout()
        self.name_label = QLabel("이름:", self)
        self.name_edit = QLineEdit(self)
        hbox1.addWidget(self.name_label)
        hbox1.addWidget(self.name_edit)

        hbox2 = QHBoxLayout()
        self.student_id_label = QLabel("학번:", self)
        self.student_id_edit = QLineEdit(self)
        hbox2.addWidget(self.student_id_label)
        hbox2.addWidget(self.student_id_edit)

        # Register 버튼을 추가합니다
        self.register_button = QPushButton("Register", self)
        self.register_button.clicked.connect(self.register_face)

        vbox.addLayout(hbox1)
        vbox.addLayout(hbox2)
        vbox.addWidget(self.register_button)

        # 타이머를 설정합니다
        self.timer = QTimer()
        self.timer.timeout.connect(self.viewCam)
        self.timer.start(20)

        self.user_id = user_id

        self.setLayout(vbox)
        self.setWindowTitle("Register")
        
        

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

                # 얼굴 영역에 라벨 표시
                left, top, right, bottom = (
                    face.left(),
                    face.top(),
                    face.right(),
                    face.bottom(),
                )
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # OpenCV 형식의 이미지를 Pixmap으로 변환합니다
            image = QImage(
                frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888
            ).rgbSwapped()
            pixmap = QPixmap.fromImage(image)

            # Pixmap을 라벨에 표시합니다
            self.label.setPixmap(pixmap)

    def register_face(self):
        name = self.name_edit.text()
        student_id = self.student_id_edit.text()

        if name and student_id:
            ret, frame = self.cap.read()
            dlib_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 얼굴 인식 수행
            faces = self.face_detector(dlib_frame)

            if len(faces) == 1:
                face = faces[0]
                landmarks = self.landmark_detector(dlib_frame, face)
                embedding = self.face_recognition_model.compute_face_descriptor(
                    dlib_frame, landmarks
                )
                np.save(f"faces/{student_id}.npy", embedding)
                # MySQL에 쿼리를 실행하여 학생 정보를 등록합니다
                self.save_student(student_id, name)
                QMessageBox.information(
                    self, "Registration Successful", "Face registered successfully!"
                )
            else:
                QMessageBox.warning(
                    self, "Registration Error", "Please make sure only one face is visible."
                )
        else:
            QMessageBox.warning(
                self, "Registration Error", "Please enter name and student ID."
            )

    def save_student(self, student_id, name):
        cursor = self.db_connection.cursor()
        sql = "INSERT INTO student (id, name, parent_id, teacher_id) VALUES (%s, %s, %s, %s)"
        values = (student_id, name, None, self.user_id)
        cursor.execute(sql, values)
        self.name_edit.clear()
        self.student_id_edit.clear()
        self.db_connection.commit()

    def closeEvent(self, event):
        event.accept()
        self.cap.release()
        self.db_connection.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Register("teacher_id_here")
    ex.show()
    sys.exit(app.exec_())