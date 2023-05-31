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
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import pymysql
from datetime import datetime
from video_thread import VideoThread

face_detector = dlib.get_frontal_face_detector()
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
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
def check_attendance(student_id, attend_type, teacher_id="12345"):

    current_date = datetime.now().strftime('%Y-%m-%d')
    sql = "INSERT INTO attendance (attend_type, date, student_id, teacher_id) VALUES (%s, %s, %s, %s)"
    cursor.execute(sql, (attend_type, current_date, student_id, teacher_id))

    # 변경사항을 커밋
    conn.commit()
# 0 - 출첵, 1 - 결석, 2 - 실종

class Attendance(QWidget):
    closed = pyqtSignal()


    def __init__(self):
        super().__init__()
        self.title = "Attendance"
        self.known_embeddings = []
        self.known_labels = []
        self.attended = []
        self.initUI()
        self.load_known_faces()

    def initUI(self):
        self.setWindowTitle(self.title)

        # video
        self.image_label = QLabel(self)
        self.image_label.resize(1174, 632)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        self.th = VideoThread(self)
        self.th.change_pixmap_signal.connect(self.update_image)
        self.th.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_attendance)
        self.timer.start(10000)

    def load_known_faces(self):
        self.known_embeddings = []
        self.known_labels = []

        npy_files = [file for file in os.listdir() if file.endswith(".npy")]
        for file in npy_files:
            embedding = np.load(file)
            label = file.replace(".npy", "")
            self.known_embeddings.append(embedding)
            self.known_labels.append(label)
            self.attended.append(False)

    def update_attendance(self):
        ret, frame = self.th.cap.read()
        if ret:
            dlib_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector(dlib_frame)
            for face in faces:
                landmarks = landmark_detector(dlib_frame, face)
                unknown_embedding = face_recognition_model.compute_face_descriptor(dlib_frame, landmarks)
                unknown_embedding = np.array(unknown_embedding)

                min_distance = 1.0
                min_distance_index = -1
                for i, known_embedding in enumerate(self.known_embeddings):
                    distance = np.linalg.norm(unknown_embedding - known_embedding)
                    if distance < min_distance:
                        min_distance = distance
                        min_distance_index = i

                if min_distance < 0.6:
                    check_attendance(self.known_labels[min_distance_index],0)
                    self.attended[min_distance_index] = True
                    QMessageBox.information(self, "알림", f"{self.known_labels[min_distance_index]} 출석체크 되었습니다.")
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
        for i in range(len(self.attended)):
            if not self.attended[i]:
                check_attendance(self.known_labels[i], 0)

        if reply == QMessageBox.Yes:
            self.close()
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
