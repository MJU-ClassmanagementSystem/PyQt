import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QTimer
import cv2
import dlib

import pymysql

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

class MyApp(QWidget):
    def __init__(self):
        super().__init__()

        # 카메라 객체를 설정합니다
        self.cap = cv2.VideoCapture(0)

        # Load models and initialize variables
        self.face_detector = dlib.get_frontal_face_detector()

        # 이미지를 표시할 라벨을 설정합니다
        self.label = QLabel(self)

        # 학생 정보를 입력할 라인 에디트를 설정합니다
        self.input_student_id = QLineEdit()
        self.input_student_name = QLineEdit()

        # 학번과 이름을 표시할 라벨을 설정합니다
        self.label_student_id = QLabel("학번:", self)
        self.label_student_name = QLabel("이름:", self)

        # 등록 버튼을 설정합니다
        self.button_register = QPushButton("Register")
        self.button_register.setEnabled(False)  # 초기에 비활성화 상태로 설정합니다
        self.button_register.clicked.connect(self.registerFace)

        # 수평 레이아웃을 생성하고 위젯을 추가합니다
        hbox_id = QHBoxLayout()
        hbox_id.addWidget(self.label_student_id)
        hbox_id.addWidget(self.input_student_id)

        hbox_name = QHBoxLayout()
        hbox_name.addWidget(self.label_student_name)
        hbox_name.addWidget(self.input_student_name)

        hbox_register = QHBoxLayout()
        hbox_register.addWidget(self.button_register)

        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addLayout(hbox_id)
        vbox.addLayout(hbox_name)
        vbox.addLayout(hbox_register)

        self.setLayout(vbox)
        self.setWindowTitle("Face Register with PyQt")

        # 타이머를 설정합니다
        self.timer = QTimer()
        self.timer.timeout.connect(self.viewCam)
        self.timer.start(1)

    def viewCam(self):
        # OpenCV를 이용하여 카메라로부터 이미지를 캡쳐합니다
        ret, frame = self.cap.read()
        dlib_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 얼굴 인식 수행
        faces = self.face_detector(dlib_frame)

        # 얼굴이 인식되었다면 등록 버튼을 활성화합니다
        if len(faces) > 0:
            self.button_register.setEnabled(True)
            # 얼굴 주위에 네모 상자 그리기
            for face in faces:
                left = face.left()
                top = face.top()
                right = face.right()
                bottom = face.bottom()
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        else:
            self.button_register.setEnabled(False)

        # OpenCV 형식의 이미지를 Pixmap으로 변환합니다
        image = QImage(
            frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888
        ).rgbSwapped()
        pixmap = QPixmap.fromImage(image)

        # Pixmap을 라벨에 표시합니다
        self.label.setPixmap(pixmap)

    def registerFace(self):
        # 학번과 이름을 가져옵니다
        student_id = self.input_student_id.text()
        student_name = self.input_student_name.text()

        # OpenCV를 이용하여 카메라로부터 이미지를 캡쳐합니다
        ret, frame = self.cap.read()

        # 이미지를 저장합니다
        cv2.imwrite(f"{student_id}_{student_name}.jpg", frame)
        save_student(student_id, student_name)
        print("Face registered!")

    def closeEvent(self, event):
        event.accept()
        self.cap.release()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
