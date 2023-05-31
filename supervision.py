import glob
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2
import dlib
import numpy as np
import tensorflow as tf
from keras.models import load_model
from video_thread import VideoThread


class Supervision(QWidget):
    def __init__(self,main_menu):
        super().__init__()
        self.main_menu = main_menu

        self.face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
        self.landmark_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        


        self.known_embeddings = []
        self.known_labels = []
        self.blink_counts = []

        self.face_detector = dlib.get_frontal_face_detector()

        # self.load_face_embeddings(['bohyun.npy'])
        self.load_all_embeddings()

        self.emotion_model = load_model('emotion_model.hdf5')
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        self.th = VideoThread(self)
        self.th.change_pixmap_signal.connect(self.update_image)
        self.th.start()

        self.initUI()

    def initUI(self):
        self.image_label = QLabel(self)
        self.image_label.resize(640, 480)

        self.main_menu_button = QPushButton("Main Menu", self)
        self.main_menu_button.clicked.connect(self.go_to_main_menu)

        # Connect the main menu button to an appropriate slot...

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.main_menu_button)

        self.setLayout(layout)
        
        
    def go_to_main_menu(self):
        self.th.stop()
        self.close()
        self.main_menu.show()


    def add_known_face_embedding(self, embedding, label):
        self.known_embeddings.append(embedding)
        self.known_labels.append(label)
        self.blink_counts.append(0)
        
        
    def load_all_embeddings(self):
        npy_files = glob.glob("faces/*.npy")
        self.load_face_embeddings(npy_files)

    # def load_face_embeddings(self, images):
    #     for image in images:
    #         person_image = cv2.imread(image)
    #         person_dlib_frame = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
    #         person_faces = self.face_detector(person_dlib_frame)
    #         person_landmarks = self.landmark_detector(person_dlib_frame, person_faces[0])
    #         person_embedding = self.face_recognition_model.compute_face_descriptor(person_dlib_frame, person_landmarks)
    #         label = image.split('0')[0]
    #         self.add_known_face_embedding(person_embedding, label)
    
    
    def load_face_embeddings(self, npy_files):
        faces_dir = 'faces'
        npy_files = [file for file in os.listdir(faces_dir) if file.endswith(".npy")]
        for file in npy_files:
            person_embedding = np.load(os.path.join(faces_dir, file))
            label = file.replace(".npy", "")
            self.add_known_face_embedding(person_embedding, label)

    @staticmethod
    def calculate_eye_aspect_ratio(eye_landmarks):
        eye_landmarks = np.array([[p.x, p.y] for p in eye_landmarks])

        horizontal_dist1 = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        horizontal_dist2 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[2])
        horizontal_length = (horizontal_dist1 + horizontal_dist2) / 2

        vertical_dist1 = np.linalg.norm(eye_landmarks[5] - eye_landmarks[1])
        vertical_dist2 = np.linalg.norm(eye_landmarks[4] - eye_landmarks[2])
        vertical_length = (vertical_dist1 + vertical_dist2) / 2

        aspect_ratio = vertical_length / horizontal_length

        return aspect_ratio

    def update_image(self, cv_img):
        dlib_frame = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        faces = self.face_detector(dlib_frame)

        for face in faces:
            landmarks = self.landmark_detector(dlib_frame, face)
            embedding = self.face_recognition_model.compute_face_descriptor(dlib_frame, landmarks)

            distances = []
            for known_embedding in self.known_embeddings:
                distance = np.linalg.norm(np.array(embedding) - np.array(known_embedding))
                distances.append(distance)

            min_distance_idx = np.argmin(distances)
            min_distance = distances[min_distance_idx]
            
            # This part includes blink detection.
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

            gray_landmarks = self.landmark_detector(gray, face)

            # Indices for eye landmarks.
            left_eye_landmarks = [gray_landmarks.part(i) for i in range(36, 42)]
            right_eye_landmarks = [gray_landmarks.part(i) for i in range(42, 48)]

            # Calculate eye aspect ratios.
            left_eye_aspect_ratio = self.calculate_eye_aspect_ratio(left_eye_landmarks)
            right_eye_aspect_ratio = self.calculate_eye_aspect_ratio(right_eye_landmarks)

            # Check for blinks.
            if left_eye_aspect_ratio < 0.3 and right_eye_aspect_ratio < 0.3:
                self.blink_counts[min_distance_idx] += 1

            # Draw eye regions.
            for point in left_eye_landmarks:
                cv2.circle(cv_img, (point.x, point.y), 1, (0, 0, 255), -1)
            for point in right_eye_landmarks:
                cv2.circle(cv_img, (point.x, point.y), 1, (0, 0, 255), -1)

            cv2.putText(cv_img, f"Blinks: {self.blink_counts[min_distance_idx]}", (face.left(), face.top() - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if min_distance <= 0.4:
                label = self.known_labels[min_distance_idx]
            else:
                label = 'Unknown'

            left, top, right, bottom = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(cv_img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(cv_img, f"{label} {min_distance}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            face_img = cv_img[top:bottom, left:right]
            if face_img.size > 0:
                gray_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

                resized_img = cv2.resize(gray_face_img, (64, 64), interpolation=cv2.INTER_AREA)
                img_array = tf.keras.preprocessing.image.img_to_array(resized_img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255
                predictions = self.emotion_model.predict(img_array)
                max_index = np.argmax(predictions[0])
                emotion = self.emotion_labels[max_index]
                emotions = {k: v for k, v in zip(self.emotion_labels, predictions[0])}

                cv2.putText(cv_img, emotion, (left, top - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

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

    def run(self):
        self.load_face_embeddings(['person1.jpg', 'person2.jpg'])
        self.show()


# app = QApplication([])
# ex = Supervision()
# ex.run()
# app.exec_()
