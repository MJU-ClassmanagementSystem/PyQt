from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox, QLabel
from analyze import MyApp
from register import Register
from attendance import Attendance

class MainMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Main Menu"
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(500, 500, 500, 200)

        self.label_user_id = QLabel("ID here!", self)

        self.register_button = QPushButton("Register", self)
        self.register_button.clicked.connect(self.open_register)

        self.attendance_button = QPushButton("Attendance", self)
        self.attendance_button.clicked.connect(self.open_attendance)

        self.analyze_button = QPushButton("Analyze", self)
        self.analyze_button.clicked.connect(self.open_analyze)


        layout = QVBoxLayout()
        layout.addWidget(self.register_button)
        layout.addWidget(self.attendance_button)
        layout.addWidget(self.analyze_button)

        self.setLayout(layout)

    def open_register(self):
        # self.hide()
        # self.register = Register()
        self.register = Register(self.user_id)
        self.register.show()

    def open_attendance(self):
        # self.hide()
        self.attendance = Attendance(self.user_id)
        self.attendance.show()
        
    def open_analyze(self):
        # self.hide()
        self.analyze = MyApp()
        self.analyze.show()

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self, 'Message', 'Are you sure you want to exit?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            print("종료되었습니다.")
            event.accept()
        else:
            event.ignore()

    def set_user_id(self, user_id):
        self.user_id = user_id
        self.label_user_id.setText("id: " + self.user_id)

if __name__ == "__main__":
    app = QApplication([])
    ex = MainMenu()
    ex.show()
    app.exec_()