import cv2
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
import numpy as np



class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.capture_frame)
        self.timer.start(1000/30)  # Capture at 30 fps

    def run(self):
        self.exec_()

    def capture_frame(self):
        ret, cv_img = self.cap.read()
        if ret:
            self.change_pixmap_signal.emit(cv_img)