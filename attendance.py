from datetime import date
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, QLabel
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import os
import cv2
import dlib
import numpy as np
from video_thread import VideoThread
import mysql.connector

face_detector = dlib.get_frontal_face_detector()
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

db_config = {
        'host' : '34.22.65.53',
        'user' : 'root',
        'password' : 'cms',
        'database' : 'cms'
}

class Attendance(QWidget):
    def __init__(self, main_menu):
        super().__init__()
        self.title = "Attendance"
        self.known_embeddings = []
        self.known_labels = []
        self.main_menu = main_menu
        self.initUI()
        self.load_known_faces()
        self.th = VideoThread(self)
        self.th.change_pixmap_signal.connect(self.update_image)
        self.th.start()

    def initUI(self):
        self.setWindowTitle(self.title)

        self.image_label = QLabel(self)
        self.image_label.resize(640, 480)

        self.attendance_table = QTableWidget(self)
        self.attendance_table.setColumnCount(2)
        self.attendance_table.setHorizontalHeaderLabels(["Name", "Attendance"])
        self.update_button = QPushButton("Update", self)
        self.update_button.clicked.connect(self.update_attendance)
        self.main_menu_button = QPushButton("Main Menu", self)
        self.main_menu_button.clicked.connect(self.go_to_main_menu)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.attendance_table)
        layout.addWidget(self.update_button)
        layout.addWidget(self.main_menu_button)


        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_attendance)
        self.timer.start(10000)
        
    def go_to_main_menu(self):
        self.th.stop()
        self.close()
        self.main_menu.show()

    def load_known_faces(self):
        self.known_embeddings = []
        self.known_labels = []

        faces_dir = 'faces'
        npy_files = [file for file in os.listdir(faces_dir) if file.endswith(".npy")]
        for file in npy_files:
            embedding = np.load(os.path.join(faces_dir, file))
            label = file.replace(".npy", "")
            self.known_embeddings.append(embedding)
            self.known_labels.append(label)

        self.attendance_table.setRowCount(len(self.known_labels))
        for i, label in enumerate(self.known_labels):
            self.attendance_table.setItem(i, 0, QTableWidgetItem(label))
            self.attendance_table.setItem(i, 1, QTableWidgetItem("X"))

    # def update_attendance(self):
    #     ret, frame = self.th.cap.read()
    #     if ret:
    #         dlib_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         faces = face_detector(dlib_frame)
    #         for face in faces:
    #             landmarks = landmark_detector(dlib_frame, face)
    #             unknown_embedding = face_recognition_model.compute_face_descriptor(dlib_frame, landmarks)
    #             unknown_embedding = np.array(unknown_embedding)

    #             min_distance = 1.0
    #             min_distance_index = -1
    #             for i, known_embedding in enumerate(self.known_embeddings):
    #                 distance = np.linalg.norm(unknown_embedding - known_embedding)
    #                 if distance < min_distance:
    #                     min_distance = distance
    #                     min_distance_index = i

    #             if min_distance < 0.6:
    #                 self.attendance_table.setItem(min_distance_index, 1, QTableWidgetItem("O"))


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
                    # student_id = self.known_labels[min_distance_index]
                    student_id = "1"
                    attend_type = 0  # 출석 상태를 나타내는 값, O: 0, X: 1
                    teacher_id = "60182995"
                    current_date = date.today().strftime("%Y-%m-%d")
                    
                    try:
                        connection = mysql.connector.connect(**db_config)
                        cursor = connection.cursor()
                        
                        # 출석 정보 저장 쿼리 실행
                        sql = "INSERT INTO attendance (attend_type, date, student_id, teacher_id) VALUES (%s, %s, %s, %s)"
                        values = (attend_type, current_date, student_id, teacher_id)
                        cursor.execute(sql, values)
                        connection.commit()
                        
                        # 출석 테이블 업데이트
                        for i in range(self.attendance_table.rowCount()):
                            if self.attendance_table.item(i, 0).text() == student_id:
                                self.attendance_table.setItem(i, 1, QTableWidgetItem("O"))
                                break
                    except mysql.connector.Error as error:
                        print(f"Error while connecting to MySQL: {error}")
                    finally:
                        if connection.is_connected():
                            cursor.close()
                            connection.close()

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    @staticmethod
    def convert_cv_qt(cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

