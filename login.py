import mysql.connector as mc
from PyQt5.QtWidgets import *
import sys
from menu import MainMenu

# MySQL 연결 설정
def connect_to_mysql():
    mydb = mc.connect(
        host="34.22.65.53",
        user="root",
        password="cms",
        database="cms"
    )
    return mydb

class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Login"
        self.initUI()
        self.main_menu = None  # MainMenu 인스턴스
        self.user_id = None  # 사용자 ID

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(500, 500, 500, 200)

        self.label_id = QLabel("Id:", self)
        self.lineEdit_id = QLineEdit(self)

        self.label_password = QLabel("Password:", self)
        self.lineEdit_password = QLineEdit(self)
        self.lineEdit_password.setEchoMode(QLineEdit.Password)

        self.button_login = QPushButton("Login", self)
        self.button_login.clicked.connect(self.login)

        layout = QVBoxLayout()
        layout.addWidget(self.label_id)
        layout.addWidget(self.lineEdit_id)
        layout.addWidget(self.label_password)
        layout.addWidget(self.lineEdit_password)
        layout.addWidget(self.button_login)

        self.setLayout(layout)

    def login(self):
        id = self.lineEdit_id.text()
        password = self.lineEdit_password.text()

        # MySQL 연결
        mydb = connect_to_mysql()
        mycursor = mydb.cursor()

        # id와 password로 인증 확인
        query = "SELECT id FROM teacher WHERE id = %s AND password = %s"
        values = (id, password)
        mycursor.execute(query, values)
        result = mycursor.fetchone()

        if result is not None:
            self.user_id = result[0]  # 사용자 ID 설정
            self.hide()
            self.main_menu = MainMenu()
            self.main_menu.set_user_id(self.user_id)  # 사용자 ID 설정 메서드 호출
            self.main_menu.show()
        else:
            QMessageBox.warning(self, "Login Failed", "Incorrect id or password")

        # 연결 종료
        mycursor.close()
        mydb.close()

    def closeEvent(self, event):
        if self.main_menu is not None:
            self.main_menu.show()  # 메인 메뉴 창 보여주기
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = LoginWindow()
    mainWindow.show()
    sys.exit(app.exec_())