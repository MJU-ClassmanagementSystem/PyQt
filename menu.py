from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from register import Register
from attendance import Attendance

class MainMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Main Menu"
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)

        self.register_button = QPushButton("Register", self)
        self.register_button.clicked.connect(self.open_register)

        self.attendance_button = QPushButton("Attendance", self)
        self.attendance_button.clicked.connect(self.open_attendance)

        layout = QVBoxLayout()
        layout.addWidget(self.register_button)
        layout.addWidget(self.attendance_button)

        self.setLayout(layout)

    def open_register(self):
        self.hide()
        self.register = Register(self)
        self.register.show()

    def open_attendance(self):
        self.hide()
        self.attendance = Attendance(self)
        self.attendance.show()

app = QApplication([])
ex = MainMenu()
ex.show()
app.exec_()