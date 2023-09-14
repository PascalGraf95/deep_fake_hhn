import sys
from PyQt6 import QtWidgets
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor, QWindow, QMovie, QFont, QIcon
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QFrame
from output import Ui_DeepFakeHHN
import cv2
import numpy as np
import time
import os

"""
python test_video_swapspecific.py --crop_size 224 --use_mask --pic_specific_path ./demo_file/titanic.png --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path ./demo_file/titanic.jpg --video_path ./demo_file/titanic_short.mp4  --output_path ./output/titanic.mp4 --temp_path ./temp_results --no_simswaplogo

"""


selected_style_sheet = "background-repeat: no-repeat;"\
                       "background-position: center;"\
                       "border: 5px solid;"\
                       "border-color: rgb(0, 200, 0); "\
                       "border-radius: 3px; " \
                       "padding-right: 0px;" \
                       " padding-left: 0px; "\
                       "padding-top: 0px; padding-bottom: 0px;"

default_style_sheet = ""


class MainWindow(QtWidgets.QMainWindow, Ui_DeepFakeHHN):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.showFullScreen()

        # region - Internal Variables -
        # region State Variables and Objects
        self.selected_clip = 0
        self.recorded_image = None
        self.movie = QMovie("./loader.gif")
        # self.background_clip = QMovie()
        self.camera = cv2.VideoCapture(0)
        self.background_clip = cv2.VideoCapture("./videos/distant_particles_loop.mp4")
        # endregion

        # region Palettes
        self._blue_palette = QPalette()
        self._blue_palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 255))

        self._green_palette = QPalette()
        self._green_palette.setColor(QPalette.ColorRole.Window, QColor(0, 255, 0))

        self._yellow_palette = QPalette()
        self._yellow_palette.setColor(QPalette.ColorRole.Window, QColor(255, 255, 0))
        # endregion

        # region Initial Label States
        hhn_logo = QPixmap("./images/logos/hhn_logo.png").scaledToHeight(150)
        self.image_hhn_logo.setPixmap(hhn_logo)
        saai_logo = QPixmap("./images/logos/saai_logo.png").scaledToHeight(150)
        self.image_saai_logo.setPixmap(saai_logo)

        self.image_scaling = 0.8
        self.image_preview_1.setMargin(5)
        preview_image_01 = QPixmap("./images/preview/01_Titanic.jpg").scaledToHeight(
            int(self.image_preview_1.height()*self.image_scaling))
        self.image_preview_1.setPixmap(preview_image_01)

        self.image_preview_2.setMargin(5)
        preview_image_02 = QPixmap("./images/preview/02_Braveheart.jpg").scaledToHeight(
            int(self.image_preview_2.height()*self.image_scaling))
        self.image_preview_2.setPixmap(preview_image_02)

        self.image_preview_3.setMargin(5)
        preview_image_03 = QPixmap("./images/preview/03_FluchDerKaribik.jpg").scaledToHeight(
            int(self.image_preview_3.height()*self.image_scaling))
        self.image_preview_3.setPixmap(preview_image_03)

        self.image_preview_4.setMargin(5)
        preview_image_04 = QPixmap("./images/preview/04_TheOffice.jpg").scaledToHeight(
            int(self.image_preview_4.height()*self.image_scaling))
        self.image_preview_4.setPixmap(preview_image_04)

        # endregion


        # endregion

        # region - Events -
        # region Clip Selection
        self.button_preview_1.clicked.connect(lambda: self.select_clip(1))
        self.button_preview_2.clicked.connect(lambda: self.select_clip(2))
        self.button_preview_3.clicked.connect(lambda: self.select_clip(3))
        self.button_preview_4.clicked.connect(lambda: self.select_clip(4))
        # endregion

        # region Other Events
        self.button_generate.clicked.connect(self.generate)
        self.tool_box.currentChanged.connect(self.change_tool_box)
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
        if idx == 1:
            self.image_preview_1.setStyleSheet(selected_style_sheet)
        else:
            self.image_preview_1.setStyleSheet(default_style_sheet)

        if idx == 2:
            self.image_preview_2.setStyleSheet(selected_style_sheet)
        else:
            self.image_preview_2.setStyleSheet(default_style_sheet)

        if idx == 3:
            self.image_preview_3.setStyleSheet(selected_style_sheet)
        else:
            self.image_preview_3.setStyleSheet(default_style_sheet)

        if idx == 4:
            self.image_preview_4.setStyleSheet(selected_style_sheet)
        else:
            self.image_preview_4.setStyleSheet(default_style_sheet)

    def generate(self):
        if self.selected_clip == 0:
            return
        if self.tool_box.currentIndex() == 0:
            self.tool_box.setEnabled(False)
            self.progress_bar.setValue(25)
            time.sleep(1)

            self.progress_bar.setValue(50)
            time.sleep(1)

            self.progress_bar.setValue(75)
            time.sleep(1)

            self.progress_bar.setValue(100)


            # 1. Switch to Output UI
            self.tool_box.setEnabled(True)
            self.tool_box_page_2.setEnabled(True)
            self.set_output_images()

            self.tool_box.setCurrentIndex(1)
            self.progress_bar.setValue(0)

            # 2. Detect Face in recorded image

            # 3. Apply Face Swap for three images of the selected scene

            # 4. Show Results

            # 5. Meanwhile apply Face Swap to the whole video in background

            # 6. Show Results
        elif self.tool_box.currentIndex() == 1:

            self.tool_box.setCurrentIndex(0)


    def change_tool_box(self):
        if self.tool_box.currentIndex() == 0:
            self.select_clip(0)
            self.button_generate.setText("Generate")
        elif self.tool_box.currentIndex() == 1:
            self.button_generate.setText("Try Again")


    def set_output_images(self):
        image_size = np.clip(int(self.label_original_1.height()), 220, 500)
        # Original Images
        image_path_01 = os.path.join(r".\images\scenes", "{:02d}".format(self.selected_clip), "01.jpg")
        original_image_01 = QPixmap(image_path_01).scaledToHeight(image_size)
        self.label_original_1.setPixmap(original_image_01)

        image_path_02 = os.path.join(r".\images\scenes", "{:02d}".format(self.selected_clip), "02.jpg")
        original_image_02 = QPixmap(image_path_02).scaledToHeight(image_size)
        self.label_original_2.setPixmap(original_image_02)

        image_path_03 = os.path.join(r".\images\scenes", "{:02d}".format(self.selected_clip), "03.jpg")
        original_image_03 = QPixmap(image_path_03).scaledToHeight(image_size)
        self.label_original_3.setPixmap(original_image_03)

        # Generated Images
        image_path_01 = os.path.join(r".\images\scenes", "{:02d}".format(self.selected_clip), "01.jpg")
        deep_fake_01 = QPixmap(image_path_01).scaledToHeight(image_size)
        self.label_deep_fake_1.setPixmap(deep_fake_01)

        image_path_02 = os.path.join(r".\images\scenes", "{:02d}".format(self.selected_clip), "02.jpg")
        deep_fake_02 = QPixmap(image_path_02).scaledToHeight(image_size)
        self.label_deep_fake_2.setPixmap(deep_fake_02)

        image_path_03 = os.path.join(r".\images\scenes", "{:02d}".format(self.selected_clip), "03.jpg")
        deep_fake_03 = QPixmap(image_path_03).scaledToHeight(image_size)
        self.label_deep_fake_3.setPixmap(deep_fake_03)
    # endregion

    # region Live Images
    def update_webcam_image(self):
        ret, image = self.camera.read()
        if ret:
            resize_ratio = np.min([self.image_webcam.size().width()/image.shape[1],
                                   self.image_webcam.size().height()/image.shape[0]])
            image = cv2.resize(image, dsize=(int(image.shape[1]*resize_ratio), int(image.shape[0]*resize_ratio)))
            self.recorded_image = image
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
