import sys
import subprocess
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QHBoxLayout, QPushButton, QWidget


class ButtonWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        hlayout = QHBoxLayout()

        btn1 = QPushButton("학생 등록")
        btn1.clicked.connect(lambda: self.run_script("button1.py"))
        btn1.setFixedSize(150, 50)

        btn2 = QPushButton("출석 체크")
        btn2.clicked.connect(lambda: self.run_script("attendance.py"))
        btn2.setFixedSize(150, 50)

        btn3 = QPushButton("수업")
        btn3.clicked.connect(lambda: self.run_script("button3.py"))
        btn3.setFixedSize(150, 50)

        layout.addWidget(btn1)
        layout.addWidget(btn2)
        layout.addWidget(btn3)

        hlayout.addStretch()  # 여백 추가
        hlayout.addLayout(layout)  # 수평 레이아웃에 수직 레이아웃 추가
        hlayout.addStretch()  # 여백 추가

        self.setLayout(hlayout)  # 수평 레이아웃 설정
        self.resize(800, 600)  # 윈도우 크기 설정

        self.setWindowTitle("Vertical Buttons")

    def run_script(self, script_path):
        subprocess.run(["python", script_path])


app = QApplication(sys.argv)
window = ButtonWindow()
window.show()

sys.exit(app.exec_())
