import os
import cv2
import dlib
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QLineEdit,
QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import pymysql
from video_thread import VideoThread

# dlib face detector
face_detector = dlib.get_frontal_face_detector()
# dlib face recognition model
face_recognition_model = dlib.face_recognition_model_v1(
    "dlib_face_recognition_resnet_model_v1.dat"
)
# dlib face landmark detector
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


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
def save_student(id, name, teacher_id="12345"):

    # 이미지 BLOB 데이터와 동적으로 입력 받은 값들을 삽입하는 SQL 쿼리
    sql = "INSERT INTO student (id, name, teacher_id) VALUES (%s, %s, %s)"

    # BLOB 데이터 삽입
    cursor.execute(sql, (id, name, teacher_id))

    # 변경사항을 커밋
    conn.commit()

class Register(QWidget):
    def __init__(self, main_meun):
        super().__init__()

        self.title = "Register"
        self.initUI()
        self.main_menu = main_meun

    def initUI(self):
        self.setWindowTitle(self.title)

        # video
        self.image_label = QLabel(self)
        self.image_label.resize(1174, 632)

        # 학생 정보를 입력할 라인 에디트를 설정합니다
        self.input_student_id = QLineEdit()
        self.input_student_name = QLineEdit()

        # 학번과 이름을 표시할 라벨을 설정합니다
        self.label_student_id = QLabel("학번:", self)
        self.label_student_name = QLabel("이름:", self)

        self.register_button = QPushButton("Register", self)
        self.register_button.clicked.connect(self.handle_register_button)
        # self.register_button.clicked.connect(self.register_face)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        hbox_id = QHBoxLayout()
        hbox_id.addWidget(self.label_student_id)
        hbox_id.addWidget(self.input_student_id)

        hbox_name = QHBoxLayout()
        hbox_name.addWidget(self.label_student_name)
        hbox_name.addWidget(self.input_student_name)
        layout.addLayout(hbox_id)
        layout.addLayout(hbox_name)

        layout.addWidget(self.register_button)

        self.setLayout(layout)

        self.th = VideoThread(self)
        self.th.change_pixmap_signal.connect(self.update_image)
        self.th.start()


    def handle_register_button(self):
        ret, frame = self.th.cap.read()
        if ret:
            dlib_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector(dlib_frame)
            if len(faces) > 0:
                if self.register_face():
                    QMessageBox.information(self, "알림", "얼굴이 저장되었습니다.")
            else:
                QMessageBox.warning(self, "알림", "얼굴이 인식되지 않았습니다.")
    def register_face(self):
        ret, frame = self.th.cap.read()
        if ret:
            dlib_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector(dlib_frame)
            for face in faces:
                landmarks = landmark_detector(dlib_frame, face)
                embedding = face_recognition_model.compute_face_descriptor(
                    dlib_frame, landmarks
                )
                embedding = np.array(embedding)  # Convert dlib vector to numpy array
                id = self.input_student_id.text().strip()
                name = self.input_student_name.text().strip()

                if id and name:  # Only proceed if the input is not empty
                    np.save(id + ".npy", embedding)
                    save_student(id, name)
                else:
                    QMessageBox.warning(self, "알림", "학번과 이름을 입력해주세요")
                    return False
        self.input_student_id.clear()
        self.input_student_name.clear()
        return True


    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    @staticmethod
    def convert_cv_qt(cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(1174, 632, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, 'Message', 'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:

            if self.th is not None:
                self.th.stop()
                self.th.wait()
            self.main_menu.show()
            event.accept()

        else:
            event.ignore()

    def stop_thread(self):
        self.th.quit()

# if __name__ == '__main__':
#     app = QApplication([])
#     ex = Attendance()
#     ex.show()
#     app.exec_()
