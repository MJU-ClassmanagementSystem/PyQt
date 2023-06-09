import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import cv2
import dlib
import numpy as np
import os
import mysql.connector
from datetime import datetime

# MySQL 연결 정보
db_config = {
    'host': '34.22.65.53',
    'user': 'root',
    'password': 'cms',
    'database': 'cms'
}



class Attendance(QWidget):
    def __init__(self, user_id):
        super().__init__()
        
        now = datetime.now()
        self.date = now.strftime("%Y-%m-%d %H:%M:%S")


        # 카메라 객체를 설정합니다
        self.cap = cv2.VideoCapture(0)

        # Load models and initialize variables
        self.init_models_and_vars()

        # 이미지를 표시할 라벨을 설정합니다
        self.label = QLabel(self)
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)

        # 타이머를 설정합니다
        self.timer = QTimer()
        self.timer.timeout.connect(self.viewCam)
        self.timer.start(20)

        self.user_id = user_id

        self.setLayout(vbox)
        self.setWindowTitle("Attendance")
        
        # 학생들을 모두 결석으로 초기화
        self.initialize_attendance()
        
    
    def initialize_attendance(self):
        # MySQL 데이터베이스에 연결합니다
        conn = mysql.connector.connect(**db_config)

        # 커서를 생성합니다
        cursor = conn.cursor()

        # 모든 학생의 목록을 가져옵니다
        query = "SELECT id FROM student where teacher_id = %s"
        cursor.execute(query, (self.get_teacher_id(),))
        students = students = cursor.fetchall()

        # 모든 학생의 출석 상태를 '결석'으로 초기화합니다
        for student in students:
            sql = "INSERT INTO attendance (attend_type, date, student_id, teacher_id) VALUES (%s, %s, %s, %s)"
            values = (1, self.date, student[0], self.user_id)

            # SQL 쿼리를 실행합니다
            cursor.execute(sql, values)

        # 변경 사항을 커밋합니다
        conn.commit()

        # 커넥션을 닫습니다
        conn.close()

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

        self.known_embeddings = []
        self.known_labels = []
        self.attended = []
        self.load_known_faces()

    def load_known_faces(self):
        faces_folder = "faces"
        npy_files = [os.path.join(faces_folder, file) for file in os.listdir(faces_folder) if file.endswith(".npy")]
        for file in npy_files:
            embedding = np.load(file)
            label = os.path.basename(file).replace(".npy", "")
            self.known_embeddings.append(embedding)
            self.known_labels.append(label)
            self.attended.append(False)

    def viewCam(self):
        # OpenCV를 이용하여 카메라로부터 이미지를 캡쳐합니다
        ret, frame = self.cap.read()
        if ret:
            dlib_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 얼굴 인식 수행
            faces = self.face_detector(dlib_frame)

            # 각 얼굴에 대해 얼굴 식별 및 감정 분석 수행
            for face in faces:
                landmarks = self.landmark_detector(dlib_frame, face)
                unknown_embedding = self.face_recognition_model.compute_face_descriptor(dlib_frame, landmarks)
                unknown_embedding = np.array(unknown_embedding)

                min_distance = 1.0
                min_distance_index = -1
                for i, known_embedding in enumerate(self.known_embeddings):
                    distance = np.linalg.norm(unknown_embedding - known_embedding)
                    if distance < min_distance:
                        min_distance = distance
                        min_distance_index = i

                if min_distance < 0.3:
                    if not self.attended[min_distance_index]:
                        self.attended[min_distance_index] = True
                        student_id = self.known_labels[min_distance_index]
                        teacher_id = self.get_teacher_id()  # 로그인 시 사용한 user_id 가져오기
                        attend_type = 0  # 출석 유형 O
                        self.check_attendance(student_id, teacher_id, attend_type)
                        QMessageBox.information(self, "알림", f"{student_id} 출석체크 되었습니다.")
                    self.attended[min_distance_index] = True

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
                    f"{self.known_labels[min_distance_index]} {min_distance}",
                    (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

            # OpenCV 형식의 이미지를 Pixmap으로 변환합니다
            image = QImage(
                frame, frame.shape[1], frame.shape[0], QImage.Format_RGB888
            ).rgbSwapped()
            pixmap = QPixmap.fromImage(image)

            # Pixmap을 라벨에 표시합니다
            self.label.setPixmap(pixmap)

    def check_attendance(self, student_id, teacher_id, attend_type):
        # 현재 시간을 가져옵니다

        # try:
        #     # MySQL 데이터베이스에 연결합니다
        #     conn = mysql.connector.connect(**db_config)

        #     # 커서를 생성합니다
        #     cursor = conn.cursor()

        #     # attendance 테이블에 데이터를 삽입하는 SQL 쿼리를 작성합니다
        #     sql = "INSERT INTO attendance (attend_type, self., student_id, teacher_id) VALUES (%s, %s, %s, %s)"
        #     values = (attend_type, self.date, student_id, teacher_id)

        #     # SQL 쿼리를 실행합니다
        #     cursor.execute(sql, values)

        #     # 변경 사항을 커밋합니다
        #     conn.commit()

        #     # 커넥션을 닫습니다
        #     conn.close()

        # except mysql.connector.Error as e:
        #     print(f"Error: {e}")
        
        try:
            # MySQL 데이터베이스에 연결합니다
            conn = mysql.connector.connect(**db_config)

            # 커서를 생성합니다
            cursor = conn.cursor()

            # 해당 학생의 오늘의 출석 데이터가 이미 있는지 확인합니다.
            query = "SELECT * FROM attendance WHERE student_id = %s AND teacher_id = %s AND DATE(date) = CURDATE()"
            cursor.execute(query, (student_id, teacher_id))

            result = cursor.fetchone()

            if result:
                # 출석 데이터가 이미 있는 경우, attendance 테이블을 업데이트 합니다.
                sql = "UPDATE attendance SET attend_type = %s, date = %s WHERE student_id = %s AND teacher_id = %s"
                values = (attend_type, self.date, student_id, teacher_id)
            else:
                # 출석 데이터가 없는 경우, attendance 테이블에 데이터를 삽입합니다.
                sql = "INSERT INTO attendance (attend_type, date, student_id, teacher_id) VALUES (%s, %s, %s, %s)"
                values = (attend_type, self.date, student_id, teacher_id)

            # SQL 쿼리를 실행합니다
            cursor.execute(sql, values)

            # 변경 사항을 커밋합니다
            conn.commit()

            # 커넥션을 닫습니다
            conn.close()

        except mysql.connector.Error as e:
            print(f"Error: {e}")


    def get_teacher_id(self):
        # 로그인 시 사용한 user_id를 가져오는 함수를 구현하세요
        # 예를 들어, 로그인할 때 저장한 user_id를 반환하도록 합니다
        # 이 예시에서는 임의로 "teacher123"을 반환하도록 설정하였습니다
        return self.user_id

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, 'Message', 'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # for i in range(len(self.attended)):
                # if not self.attended[i]:
                    # check_attendance(self.known_labels[i], 1)
            self.close()
            event.accept()
            self.cap.release()
        else:
            event.ignore()



# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     ex = MyApp()
#     ex.show()
#     sys.exit(app.exec_())