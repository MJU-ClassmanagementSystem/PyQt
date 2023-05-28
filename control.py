# control.py

from PyQt5.QtWidgets import QApplication
from register import Register
from attendance import Attendance
from menu import MainMenu

def main():
    app = QApplication([])
    ex = MainMenu()
    ex.show()
    app.exec_()

if __name__ == "__main__":
    main()
