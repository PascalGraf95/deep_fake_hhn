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

import torch
import fractions
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from simswap.models.models import create_model
from simswap.options.test_options import TestOptions
from simswap.insightface_func.face_detect_crop_multi import Face_detect_crop
from simswap.util.videoswap_specific import video_swap
import os

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class FaceDetectionOptions:
    name = "people"
    gpu_ids = "0"
    checkpoints_dir = "./checkpoints"
    norm = "batch"
    data_type = "32"
    verbose = False
    fp16 = False
    local_rank = 0

    batch_size = 8
    load_size = 1024
    final_size = 512
    label_nc = 0
    input_nc = 3
    output_nc = 3

    dataroot = './datasets/cityscapes/'
    resize_or_crop = 'scale_width'
    n_threads = 2
    max_dataset_size = float("inf")

    display_winsize = 512

    netG = 'global'
    latent_size = 512
    ngf = 64
    n_downsample_global = 3
    n_blocks_global = 6
    n_blocks_local = 3
    n_local_enhancers = 1
    niter_fix_global = 0

    feat_num = 3
    n_dowmsample_E = 4
    nef = 16
    n_clusters = 10
    image_size = 224
    norm_G = 'spectralspadesynchbatch3x3'
    semantic_nc = 3

    ntest = float("inf")
    results_dir = "./results/"
    aspect_ratio = 1.0
    phase = "test"
    which_epoch = "latest"
    how_many = 50
    cluster_path = 'features_clustered_010.npy'
    arc_path = 'arcface_model/arcface_checkpoint.tar'
    pic_a_path = 'G:/swap_data/ID/elon-musk-hero-image.jpeg'
    pic_b_path = './demo_file/multi_people.jpg'
    pic_specific_path = './crop_224/zrf.jpg'
    multispecific_dir = './demo_file/multispecific'
    video_path = 'G:/swap_data/video/HSB_Demo_Trim.mp4'
    temp_path = './temp_results'
    output_path = './output/'
    id_thres = 0.03
    crop_size = 224
    is_train = False
    use_mask = True


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
            self.detect_face_in_webcam_image()

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

    # region Simswap Functions
    def detect_face_in_webcam_image(self):
        opt = FaceDetectionOptions()
        target_path = opt.pic_specific_path
        crop_size = opt.crop_size
        torch.nn.Module.dump_patches = True

        # Create Model and set to Evaluation Mode
        model = create_model(opt)
        model.eval()

        # Create Face Detection Model
        app = Face_detect_crop(name='antelope', root='.simswap/insightface_func/models')
        app.prepare(ctx_id=0, det_size=(640, 640), mode='None')

        # Detect face in webcam image and extract features
        with torch.no_grad():
            source_face_image, _ = app.get(self.image_webcam, crop_size)
            source_face_image_pil = Image.fromarray(cv2.cvtColor(source_face_image[0], cv2.COLOR_BGR2RGB))
            source_face_image = transformer_Arcface(source_face_image_pil)
            source_face = source_face_image.view(-1, source_face_image.shape[0],
                                                 source_face_image.shape[1],
                                                 source_face_image.shape[2])

            # Convert numpy to tensor
            source_face = source_face.cuda()

            # Create latent id
            source_image_downsample = F.interpolate(source_face, size=(112, 112))
            latend_source_id = model.netArc(source_image_downsample)
            latend_source_id = F.normalize(latend_source_id, p=2, dim=1)

        # Detect the specific person to be swapped in the provided image
        with torch.no_grad():
            target_face_whole = cv2.imread(target_path)
            target_face_align_crop, _ = app.get(target_face_whole, crop_size)
            target_face_align_crop_pil = Image.fromarray(cv2.cvtColor(target_face_align_crop[0], cv2.COLOR_BGR2RGB))
            target_face = transformer_Arcface(target_face_align_crop_pil)
            target_face = target_face.view(-1, target_face.shape[0], target_face.shape[1], target_face.shape[2])
            target_face = target_face.cuda()
            target_face_downsample = F.interpolate(target_face, size=(112, 112))
            target_face_id_nonorm = model.netArc(target_face_downsample)

        # Given the source and target person ids, swap faces in the provided video
        with torch.no_grad():
            video_swap(opt.video_path, latend_source_id, target_face_id_nonorm, opt.id_thres,
                       model, app, opt.output_path, temp_results_dir=opt.temp_path, no_simswaplogo=True,
                       use_mask=opt.use_mask, crop_size=crop_size)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == '__main__':
    main()
