import cv2
from PyQt5.QtCore import QThread, pyqtSignal

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(object)

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while True:
            ret, cv_img = self.cap.read()
            if ret:
                self.change_pixmap_signal.emit(cv_img)
            else:
                print("Unable to access the camera.")
