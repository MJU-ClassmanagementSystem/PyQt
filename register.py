import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import cv2
import dlib
import numpy as np
import tensorflow as tf
from keras.models import load_model


class FaceRecognitionLogic:
    def __init__(self):
        # 카메라 객체를 설정합니다
        self.cap = cv2.VideoCapture(0)

        # 카메라 해상도 설정
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Load models and initialize variables
        self.init_models_and_vars()

    def init_models_and_vars(self):
        # 얼굴 인식 모델 로드
        self.face_recognition_model = dlib.face_recognition_model_v1(
            "dlib_face_recognition_resnet_model_v1.dat"
        )
        # 얼굴 랜드마크 모델 로드
        self.landmark_detector = dlib.shape_predictor(
            "shape_predictor_68_face_landmarks.dat"
        )

        self.known_embeddings = []
        self.known_labels = []
        self.blink_counts = []

        # 얼굴 인식을 위한 Dlib face detector 생성
        self.face_detector = dlib.get_frontal_face_detector()

        # Person 1의 얼굴 임베딩 등록
        images = ["person1.jpg", "person2.jpg"]
        for image in images:
            person_image = cv2.imread(image)
            person_dlib_frame = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
            person_faces = self.face_detector(person_dlib_frame)
            person_landmarks = self.landmark_detector(
                person_dlib_frame, person_faces[0]
            )
            person_embedding = self.face_recognition_model.compute_face_descriptor(
                person_dlib_frame, person_landmarks
            )
            label = image.split("0")[0]
            self.add_known_face_embedding(person_embedding, label)
            self.blink_counts.append(0)

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

    def process_frame(self):
        # OpenCV를 이용하여 카메라로부터 이미지를 캡쳐합니다
        ret, frame = self.cap.read()
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

        # 눈 깜빡임 검출
        for face in faces:
            landmarks = self.landmark_detector(dlib_frame, face)
            left_eye_landmarks = landmarks.parts()[36:42]
            right_eye_landmarks = landmarks.parts()[42:48]

            left_eye_aspect_ratio = self.calculate_eye_aspect_ratio(left_eye_landmarks)
            right_eye_aspect_ratio = self.calculate_eye_aspect_ratio(
                right_eye_landmarks
            )

            average_eye_aspect_ratio = (
                left_eye_aspect_ratio + right_eye_aspect_ratio
            ) / 2

            # If the average eye aspect ratio is less than 0.2, then it is a blink
            if average_eye_aspect_ratio < 0.2:
                self.blink_counts[min_distance_idx] += 1

        # Blink results 출력
        for i, blink_count in enumerate(self.blink_counts):
            cv2.putText(
                frame,
                f"{self.known_labels[i]} blinked {blink_count} times",
                (10, 30 * (i + 1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        return frame

    def stop(self):
        # Release the video capture object
        self.cap.release()

    def __del__(self):
        self.stop()


class FaceRecognitionUI(QWidget):
    def __init__(self):
        super().__init__()

        # 로직 클래스를 초기화합니다.
        self.logic = FaceRecognitionLogic()

        # UI를 설정합니다.
        self.init_ui()

    def init_ui(self):
        # UI 설정
        self.setWindowTitle("Face Recognition App")
        self.setGeometry(100, 100, self.logic.width, self.logic.height)
        self.image_label = QLabel(self)
        self.image_label.resize(self.logic.width, self.logic.height)
        self.start_button = QPushButton("Start", self)
        self.start_button.move(0, 480)
        self.start_button.clicked.connect(self.start_clicked)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(60)

    def start_clicked(self):
        if self.start_button.text() == "Start":
            # Start the timer
            self.timer.start()
            self.start_button.setText("Stop")
        else:
            # Stop the timer
            self.timer.stop()
            self.start_button.setText("Start")

    def update_frame(self):
        frame = self.logic.process_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.logic.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = FaceRecognitionUI()
    ex.show()
    sys.exit(app.exec_())
