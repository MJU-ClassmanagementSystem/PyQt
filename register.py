from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import os
import cv2
import dlib
import numpy as np
from video_thread import VideoThread

face_detector = dlib.get_frontal_face_detector()
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class Register(QWidget):
    def __init__(self,main_menu):
        super().__init__()
        self.title = "Register"
        self.main_menu = main_menu
        self.initUI()
        self.th = VideoThread(self)
        self.th.change_pixmap_signal.connect(self.update_image)
        self.th.start()

    def initUI(self):
        self.setWindowTitle(self.title)

        self.image_label = QLabel(self)
        self.image_label.resize(640, 480)

        self.name_edit = QLineEdit(self)
        self.register_button = QPushButton("Register", self)
        self.register_button.clicked.connect(self.register_face)
        self.main_menu_button = QPushButton("Main Menu", self)
        self.main_menu_button.clicked.connect(self.go_to_main_menu)


        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.name_edit)
        layout.addWidget(self.register_button)
        layout.addWidget(self.main_menu_button)


        self.setLayout(layout)
    
    def go_to_main_menu(self):
        self.close()
        self.main_menu.show()
        
    def closeEvent(self, event):
        self.th.stop()  # Assume that you have a method to stop the thread in your VideoThread class
        event.accept()  # Let the window close


    def register_face(self):
        ret, frame = self.th.cap.read()
        if ret:
            dlib_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector(dlib_frame)
            for face in faces:
                landmarks = landmark_detector(dlib_frame, face)
                embedding = face_recognition_model.compute_face_descriptor(dlib_frame, landmarks)
                embedding = np.array(embedding)
                label = self.name_edit.text().strip()

                if label:
                    np.save(label + ".npy", embedding)
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Information)
                    msg.setText("Success")
                    msg.setInformativeText('You have been registered.')
                    msg.setWindowTitle("Success")
                    msg.exec_()

                else:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Critical)
                    msg.setText("Error")
                    msg.setInformativeText('Please enter a name.')
                    msg.setWindowTitle("Error")
                    msg.exec_()

        self.name_edit.clear()

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
