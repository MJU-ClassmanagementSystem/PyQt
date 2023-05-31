from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from register import Register
from attendance import Attendance
from supervision import Supervision

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

        self.supervision_button = QPushButton("Supervision", self)
        self.supervision_button.clicked.connect(self.open_supervision)
        
        layout = QVBoxLayout()
        layout.addWidget(self.register_button)
        layout.addWidget(self.attendance_button)
        layout.addWidget(self.supervision_button)

        self.setLayout(layout)

    def open_register(self):
        self.hide()
        self.register = Register(self)
        self.register.show()

    def open_attendance(self):
        self.hide()
        self.attendance = Attendance(self)
        self.attendance.show()

    def open_supervision(self):
        self.hide()
        self.supervision = Supervision(self)
        self.supervision.show()

app = QApplication([])
ex = MainMenu()
ex.show()
app.exec_()