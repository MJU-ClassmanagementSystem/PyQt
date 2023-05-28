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
    QTableWidget,
    QTableWidgetItem,
    QLineEdit,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

# dlib face detector
face_detector = dlib.get_frontal_face_detector()
# dlib face recognition model
face_recognition_model = dlib.face_recognition_model_v1(
    "dlib_face_recognition_resnet_model_v1.dat"
)
# dlib face landmark detector
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while True:
            ret, cv_img = self.cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
            else:
                print("Unable to access the camera.")


class Attendance(QWidget):
    def __init__(self):
        super().__init__()

        self.title = "Attendance"
        self.known_embeddings = []
        self.known_labels = []
        self.initUI()
        self.load_known_faces()

    def initUI(self):
        self.setWindowTitle(self.title)

        # video
        self.image_label = QLabel(self)
        self.image_label.resize(640, 480)

        self.name_edit = QLineEdit(self)
        self.register_button = QPushButton("Register", self)
        self.register_button.clicked.connect(self.register_face)

        self.attendance_table = QTableWidget(self)
        self.attendance_table.setColumnCount(2)  # Add this line
        self.attendance_table.setHorizontalHeaderLabels(["Name", "Attendance"])
        self.update_button = QPushButton("Update", self)
        self.update_button.clicked.connect(self.update_attendance)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.name_edit)
        layout.addWidget(self.register_button)
        layout.addWidget(self.attendance_table)
        layout.addWidget(self.update_button)

        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_attendance)
        self.timer.start(10000)  # update attendance every 10 seconds

        self.th = VideoThread(self)
        self.th.change_pixmap_signal.connect(self.update_image)
        self.th.start()

    def load_known_faces(self):
        self.known_embeddings = []
        self.known_labels = []

        npy_files = [file for file in os.listdir() if file.endswith(".npy")]
        for file in npy_files:
            embedding = np.load(file)
            label = file.replace(".npy", "")
            self.known_embeddings.append(embedding)
            self.known_labels.append(label)

        self.attendance_table.setRowCount(len(self.known_labels))
        for i, label in enumerate(self.known_labels):
            self.attendance_table.setItem(i, 0, QTableWidgetItem(label))
            self.attendance_table.setItem(i, 1, QTableWidgetItem("X"))

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
                label = self.name_edit.text().strip()

                if label:  # Only proceed if the input is not empty
                    np.save(label + ".npy", embedding)
                    if (
                        label in self.known_labels
                    ):  # Check if the label is already registered
                        index = self.known_labels.index(label)
                        self.known_embeddings[
                            index
                        ] = embedding  # Update the embedding if the label is already registered
                    else:
                        self.known_embeddings.append(embedding)
                        self.known_labels.append(label)
                        self.attendance_table.insertRow(
                            self.attendance_table.rowCount()
                        )
                        self.attendance_table.setItem(
                            self.attendance_table.rowCount() - 1,
                            0,
                            QTableWidgetItem(label),
                        )
                        self.attendance_table.setItem(
                            self.attendance_table.rowCount() - 1,
                            1,
                            QTableWidgetItem("X"),
                        )

        self.name_edit.clear()

    def update_attendance(self):
        ret, frame = self.th.cap.read()
        if ret:
            dlib_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector(dlib_frame)
            for face in faces:
                landmarks = landmark_detector(dlib_frame, face)
                unknown_embedding = face_recognition_model.compute_face_descriptor(
                    dlib_frame, landmarks
                )
                unknown_embedding = np.array(
                    unknown_embedding
                )  # Convert dlib vector to numpy array

                min_distance = 1.0
                min_distance_index = -1
                for i, known_embedding in enumerate(self.known_embeddings):
                    distance = np.linalg.norm(unknown_embedding - known_embedding)
                    if distance < min_distance:
                        min_distance = distance
                        min_distance_index = i

                if min_distance < 0.6:
                    self.attendance_table.setItem(
                        min_distance_index, 1, QTableWidgetItem("O")
                    )

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
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


app = QApplication([])
ex = Attendance()
ex.show()
app.exec_()
