import sys
from PyQt6 import QtWidgets
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor, QWindow, QMovie, QFont
from PyQt6.QtCore import QTimer
from output import Ui_DeepFakeHHN
import cv2
import numpy as np
import time


class MainWindow(QtWidgets.QMainWindow, Ui_DeepFakeHHN):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # region - Internal Variables -
        # region State Variables and Objects
        self.movie = QMovie("./loader.gif")
        # self.background_clip = QMovie()
        self.camera = cv2.VideoCapture(0)
        self.background_clip = cv2.VideoCapture("./videos/distant_particles_loop.mp4")
        # endregion

        # region Initial Label States
        hhn_logo = QPixmap("./images/hhn_logo.png").scaledToWidth(200)
        self.label_hhn_logo.setPixmap(hhn_logo)
        # endregion

        # region Palettes
        self._blue_palette = QPalette()
        self._blue_palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 255))

        self._green_palette = QPalette()
        self._green_palette.setColor(QPalette.ColorRole.Window, QColor(0, 255, 0))

        self._yellow_palette = QPalette()
        self._yellow_palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 0))
        # endregion
        # endregion

        # region - Events -
        # region Clip Selection
        self.button_preview_1.clicked.connect(lambda: self.select_clip(0))
        self.button_preview_2.clicked.connect(lambda: self.select_clip(1))
        self.button_preview_3.clicked.connect(lambda: self.select_clip(2))
        self.button_preview_4.clicked.connect(lambda: self.select_clip(3))

        # endregion
        # endregion

        # region - Timers and Live Images -
        # Setup Live Camera Image
        self.live_image_timer = QTimer()
        self.live_image_timer.timeout.connect(self.update_webcam_image)
        self.live_image_timer.start(30)

        # self.background_image_timer = QTimer()
        # self.background_image_timer.timeout.connect(self.update_background_image)
        # self.background_image_timer.start(30)
        # endregion

    # region Button Events
    def select_clip(self, idx):
        pass
    # endregion

    # region Live Images
    def update_webcam_image(self):
        ret, image = self.camera.read()
        if ret:
            resize_ratio = np.min([self.image_webcam.size().width()/image.shape[1],
                                   self.image_webcam.size().height()/image.shape[0]])
            image = cv2.resize(image, dsize=(int(image.shape[1]*resize_ratio), int(image.shape[0]*resize_ratio)))
            convert = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_BGR888)
            self.image_webcam.setPixmap(QPixmap.fromImage(convert))

    def update_background_image(self):
        ret, image = self.background_clip.read()
        cv2.imwrite("./images/background_image.png", image)
        stylesheet = 'background-image: url("./images/background_image.png");'
        self.centralwidget.setStyleSheet(stylesheet)

    # endregion


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == '__main__':
    main()