import sys
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow


class VideoWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Window")

        # QLabel을 사용하여 영상을 표시할 위젯 생성
        self.label = QLabel(self)
        self.setCentralWidget(self.label)

        # 카메라 캡처를 위한 VideoCapture 객체 생성
        self.cap = cv2.VideoCapture(0)

        # 얼굴 인식을 위한 Haar Cascade 분류기 로드
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # QTimer를 사용하여 프레임 업데이트 설정
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # 얼굴 저장을 위한 카운터 및 경로 설정
        self.sample_count = 0
        self.teacher_id = "teacher_id"
        self.student_id = "student_id"

    def update_frame(self):
        # 카메라에서 프레임 읽기
        ret, frame = self.cap.read()

        if ret:
            # 프레임을 그레이스케일로 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 얼굴 검출
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5
            )

            for (x, y, w, h) in faces:
                # 얼굴 주변에 사각형 그리기
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # 얼굴 이미지 저장
                self.sample_count += 1

                cv2.imwrite(
                    f'./faces/{self.student_id}.{self.sample_count}.jpg',
                    gray[y:y+h, x:x+w]
                )

                # 100장의 얼굴을 찍으면 종료
                if self.sample_count == 100:
                    self.cap.release()
                    self.timer.stop()

            # OpenCV 이미지를 QImage로 변환
            h, w, _ = frame.shape
            q_image = QImage(
                frame.data, w, h, QImage.Format_RGB888
            ).scaled(640, 480, Qt.KeepAspectRatio)

            # QLabel에 QImage를 설정하여 영상 표시
            self.label.setPixmap(QPixmap.fromImage(q_image))

    def closeEvent(self, event):
        # 창이 닫힐 때 카메라 캡처 및 타이머 정지
        self.cap.release()
        self.timer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    video_window = VideoWindow()
    video_window.show()
    sys.exit(app.exec_())

