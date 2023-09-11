import sys
from PyQt6 import QtWidgets
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor, QWindow, QMovie, QFont, QIcon
from PyQt6.QtCore import QTimer, Qt
from output import Ui_DeepFakeHHN
import cv2
import numpy as np
import time

"""
python test_video_swapspecific.py --crop_size 224 --use_mask --pic_specific_path ./demo_file/titanic.png --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path ./demo_file/titanic.jpg --video_path ./demo_file/titanic_short.mp4  --output_path ./output/titanic.mp4 --temp_path ./temp_results --no_simswaplogo

"""





default_button_stylesheet = "background-repeat: no-repeat;"\
                            "background-position: center;"\
                            "border: 5px solid;"\
                            "border-color: rgb(46, 103, 156); "\
                            "border-radius: 3px; " \
                            "padding-right: 10px;" \
                            " padding-left: 10px; "\
                            "padding-top: 5px; padding-bottom: 5px;"


class MainWindow(QtWidgets.QMainWindow, Ui_DeepFakeHHN):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        # region - Internal Variables -
        # region State Variables and Objects
        self.selected_clip = 0
        self.movie = QMovie("./loader.gif")
        # self.background_clip = QMovie()
        self.camera = cv2.VideoCapture(0)
        self.background_clip = cv2.VideoCapture("./videos/distant_particles_loop.mp4")
        # endregion

        # region Initial Label States
        hhn_logo = QPixmap("./images/hhn_logo.png").scaledToWidth(200)
        self.label_hhn_logo.setPixmap(hhn_logo)

        # setting image to the button
        self.button_preview_1.setStyleSheet("background-image : url(images/01_Titanic.jpg); " + default_button_stylesheet)

        # self.button_preview_1.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.button_preview_2.setStyleSheet("background-image : url(images/02_Braveheart.jpg);" + default_button_stylesheet)
        self.button_preview_3.setStyleSheet("background-image : url(images/03_FluchDerKaribik.jpg);" + default_button_stylesheet)
        self.button_preview_4.setStyleSheet("background-image : url(images/04_TheOffice.jpg);" + default_button_stylesheet)
        # self.button_preview_1.setIcon(QIcon('images/01_Titanic.jpg'))
        self.button_preview_1.setText("")
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
        self.selected_clip = idx
        """
        if idx == 0:
            self.button_preview_1.setStyleSheet("background-image : url(images/01_Titanic.jpg); "
                                                "background-repeat: no-repeat;"
                                                "background-position: center;"
                                                "background-color: green")
        else:
            self.button_preview_1.setStyleSheet("background-image : url(images/01_Titanic.jpg); "
                                                "background-repeat: no-repeat;"
                                                "background-position: center;"
                                                "background-color: red")
        """
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
