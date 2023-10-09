import glob
import sys
from PyQt6 import QtWidgets
from PyQt6.QtGui import QImage, QPixmap, QPalette, QColor, QWindow, QMovie, QFont, QIcon
from PyQt6.QtMultimedia import QSoundEffect
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QFrame
from tqdm import tqdm

from output import Ui_DeepFakeHHN
import cv2
import numpy as np
import time
import os
from moviepy.editor import AudioFileClip, VideoFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import shutil
import torch
import fractions
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from simswap.models.models import create_model
from simswap.util.norm import SpecificNorm
from simswap.insightface_func.face_detect_crop_multi import Face_detect_crop
# from simswap.util.videoswap_specific import video_swap
import os
from simswap.parsing_model.model import BiSeNet
from reenactment import depth
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
from scipy.spatial import ConvexHull
import reenactment.modules.generator as GEN
from reenactment.modules.keypoint_detector import KPDetector
from collections import OrderedDict
import yaml
from reenactment.sync_batchnorm import DataParallelWithCallback
from reenactment.animate import normalize_kp
from crop_video import process_video
import subprocess
import shlex


mse = torch.nn.MSELoss().cuda()
spNorm = SpecificNorm()

transformer_Arcface = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def _to_tensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)


class FaceDetectionOptions:
    name = "224"
    gpu_ids = "0"
    checkpoints_dir = "./simswap/checkpoints"
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
    arc_path = 'simswap/arcface_model/arcface_checkpoint.tar'
    temp_path = './simswap/temp_results'
    output_path = './output/'
    id_thres = 0.05
    crop_size = 224
    is_train = False
    use_mask = True


"""
python test_video_swapspecific.py --crop_size 224 --use_mask --pic_specific_path ./demo_file/titanic.png --name people --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path ./demo_file/titanic.jpg --video_path ./demo_file/titanic_short.mp4  --output_path ./output/titanic.mp4 --temp_path ./temp_results --no_simswaplogo


python test_wholeimage_swapspecific.py --name people --pic_specific_path demo_file/khaleesi.jpg --crop_size 224 --Arc_path arcface_model/arcface_checkpoint.tar --pic_a_path demo_file/ich.jpg --pic_b_path demo_file/khaleesi.jpg --output_path output/ 
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

        # region - Bug Fix -
        self.tool_box.setCurrentIndex(1)
        time.sleep(0.1)
        self.tool_box.setCurrentIndex(0)
        # endregion

        # region - Internal Variables -
        # Simswap Objects
        self.opt = None
        self.face_swap_model = None
        self.face_det_model = None
        self.mask_net = None
        self.force_rebuild = False
        # Reenactment Objects
        self.depth_encoder = None
        self.depth_decoder = None
        # Encoded Faces & Images
        self.source_face = None
        self.source_id = None
        self.target_face = None
        self.target_id = None
        # region State Variables and Objects
        self.selected_clip = 0
        self.selected_input = 0
        self.recorded_image = None
        self.loading_animation = QMovie("./loader.gif")
        self.your_face_here = cv2.imread("./images/logos/your_face_here.png", cv2.IMREAD_UNCHANGED)
        self.marker_factor = 0.1
        self.your_face_here = cv2.resize(self.your_face_here, (0, 0), fx=self.marker_factor, fy=self.marker_factor)
        self.video_freq = 30
        self.deep_fake_video = None
        self.video_timestamp_list = []
        # self.background_clip = QMovie()
        self.camera = cv2.VideoCapture(0)
        self.record_mode = False
        self.recording_path = "temp_recording"
        # endregion

        # region Model Initialization
        self.initialize_models()
        # endregion

        # region Initial Label States
        header = QPixmap("./images/logos/Header.png").scaledToWidth(self.label_title.width())
        self.label_title.setPixmap(header)

        self.image_scaling = 0.7
        for idx, preview_image in enumerate([self.image_preview_1, self.image_preview_2, self.image_preview_3,
                                             self.image_preview_4, self.image_preview_5, self.image_preview_6,
                                             self.image_preview_7, self.image_preview_8, self.image_preview_9]):
            preview_image.setMargin(5)
            preview_image_pixmap = QPixmap("./images/preview/{:02d}.jpg".format(idx+1)).scaledToHeight(
                int(self.image_preview_1.height()*self.image_scaling))
            preview_image.setPixmap(preview_image_pixmap)

        for idx, input_image in enumerate([self.image_input_1, self.image_input_2, self.image_input_3,
                                             self.image_input_4, self.image_input_5, self.image_input_6,
                                             self.image_input_7, self.image_input_8, self.image_input_9]):
            input_image.setMargin(5)
            input_image_pixmap = QPixmap("./images/input/{:02d}.jpg".format(idx+1)).scaledToHeight(
                int(self.image_preview_1.height()*self.image_scaling))
            input_image.setPixmap(input_image_pixmap)
        # endregion
        # endregion

        # region - Events -
        # region Clip Selection
        self.button_preview_1.clicked.connect(lambda: self.select_clip(1))
        self.button_preview_2.clicked.connect(lambda: self.select_clip(2))
        self.button_preview_3.clicked.connect(lambda: self.select_clip(3))
        self.button_preview_4.clicked.connect(lambda: self.select_clip(4))
        self.button_preview_5.clicked.connect(lambda: self.select_clip(5))
        self.button_preview_6.clicked.connect(lambda: self.select_clip(6))
        self.button_preview_7.clicked.connect(lambda: self.select_clip(7))
        self.button_preview_8.clicked.connect(lambda: self.select_clip(8))
        self.button_preview_9.clicked.connect(lambda: self.select_clip(9))
        # endregion

        # region Input Selection
        self.button_face_1.clicked.connect(lambda: self.select_input(1))
        self.button_face_2.clicked.connect(lambda: self.select_input(2))
        self.button_face_3.clicked.connect(lambda: self.select_input(3))
        self.button_face_4.clicked.connect(lambda: self.select_input(4))
        self.button_face_5.clicked.connect(lambda: self.select_input(5))
        self.button_face_6.clicked.connect(lambda: self.select_input(6))
        self.button_face_7.clicked.connect(lambda: self.select_input(7))
        self.button_face_8.clicked.connect(lambda: self.select_input(8))
        self.button_face_9.clicked.connect(lambda: self.select_input(9))
        # endregion

        # region Other Events
        self.button_generate.clicked.connect(self.generate)
        self.tool_box.currentChanged.connect(self.change_tool_box)
        self.tab_widget_input.currentChanged.connect(self.change_tabs)
        self.combo_box_model.currentIndexChanged.connect(self.change_combo_model)
        self.push_button_play.clicked.connect(self.replay_video)
        # endregion
        # endregion

        # region - Timers and Live Images -
        # Setup Live Camera Image
        self.live_image_timer = QTimer()
        self.live_image_timer.timeout.connect(self.update_webcam_image)
        self.live_image_timer.start(30)

        # Setup Deep Fake Result Video
        self.deep_fake_video_timer = QTimer()
        self.deep_fake_video_timer.timeout.connect(self.update_fake_video_image)

        # Setup record
        self.stop_recording_timer = QTimer()
        self.stop_recording_timer.timeout.connect(self.stop_recording)
        # endregion

        # ToDo: Fix Video Audio
        # ToDo: Other Inception Video
        # ToDo: Better Reenactment Images
        # ToDo: Img Size Output
        # ToDo: GIVE AWAY?!
        # ToDo: Other Trump Video
        # ToDo: Poster
        # ToDo: Progress-Bar for Reenactment

    # region Initialization
    def initialize_models(self):
        if not self.opt:
            torch.nn.Module.dump_patches = True
            self.opt = FaceDetectionOptions()
        if not self.face_swap_model or self.force_rebuild:
            self.face_swap_model = create_model(self.opt)
            self.face_swap_model.eval()
            self.force_rebuild = False
        if not self.face_det_model:
            self.face_det_model = Face_detect_crop(name='antelope')
            self.face_det_model.prepare(ctx_id=0, det_size=(640, 640), mode='None')
        if not self.mask_net and self.opt.use_mask:
            n_classes = 19
            self.mask_net = BiSeNet(n_classes=n_classes)
            self.mask_net.cuda()
            model_path = os.path.join('./simswap/parsing_model/checkpoint', '79999_iter.pth')
            self.mask_net.load_state_dict(torch.load(model_path))
            self.mask_net.eval()
        if not self.depth_decoder:
            self.depth_encoder = depth.ResnetEncoder(18, False)
            self.depth_decoder = depth.DepthDecoder(num_ch_enc=self.depth_encoder.num_ch_enc, scales=range(4))
            loaded_dict_enc = torch.load('reenactment/depth/models/depth_face_model/encoder.pth')
            loaded_dict_dec = torch.load('reenactment/depth/models/depth_face_model/depth.pth')
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.depth_encoder.state_dict()}
            self.depth_encoder.load_state_dict(filtered_dict_enc)
            self.depth_decoder.load_state_dict(loaded_dict_dec)
            self.depth_encoder.eval()
            self.depth_decoder.eval()

            self.depth_encoder.cuda()
            self.depth_decoder.cuda()
    # endregion

    # region Events
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
        if idx == 5:
            self.image_preview_5.setStyleSheet(selected_style_sheet)
        else:
            self.image_preview_5.setStyleSheet(default_style_sheet)
        if idx == 6:
            self.image_preview_6.setStyleSheet(selected_style_sheet)
        else:
            self.image_preview_6.setStyleSheet(default_style_sheet)
        if idx == 7:
            self.image_preview_7.setStyleSheet(selected_style_sheet)
        else:
            self.image_preview_7.setStyleSheet(default_style_sheet)
        if idx == 8:
            self.image_preview_8.setStyleSheet(selected_style_sheet)
        else:
            self.image_preview_8.setStyleSheet(default_style_sheet)
        if idx == 9:
            self.image_preview_9.setStyleSheet(selected_style_sheet)
        else:
            self.image_preview_9.setStyleSheet(default_style_sheet)

    def select_input(self, idx):
        self.selected_input = idx
        if idx == 1:
            self.image_input_1.setStyleSheet(selected_style_sheet)
        else:
            self.image_input_1.setStyleSheet(default_style_sheet)
        if idx == 2:
            self.image_input_2.setStyleSheet(selected_style_sheet)
        else:
            self.image_input_2.setStyleSheet(default_style_sheet)
        if idx == 3:
            self.image_input_3.setStyleSheet(selected_style_sheet)
        else:
            self.image_input_3.setStyleSheet(default_style_sheet)
        if idx == 4:
            self.image_input_4.setStyleSheet(selected_style_sheet)
        else:
            self.image_input_4.setStyleSheet(default_style_sheet)
        if idx == 5:
            self.image_input_5.setStyleSheet(selected_style_sheet)
        else:
            self.image_input_5.setStyleSheet(default_style_sheet)
        if idx == 6:
            self.image_input_6.setStyleSheet(selected_style_sheet)
        else:
            self.image_input_6.setStyleSheet(default_style_sheet)
        if idx == 7:
            self.image_input_7.setStyleSheet(selected_style_sheet)
        else:
            self.image_input_7.setStyleSheet(default_style_sheet)
        if idx == 8:
            self.image_input_8.setStyleSheet(selected_style_sheet)
        else:
            self.image_input_8.setStyleSheet(default_style_sheet)
        if idx == 9:
            self.image_input_9.setStyleSheet(selected_style_sheet)
        else:
            self.image_input_9.setStyleSheet(default_style_sheet)

    def generate(self):
        self.progress_bar.setValue(0)
        self.initialize_models()
        self.label_deep_fake_video.clear()
        self.deep_fake_video = None
        if self.selected_clip == 0:
            return
        if self.tab_widget_input.currentIndex() == 1 and self.selected_input == 0:
            return
        if self.tab_widget_input.currentIndex() == 1 and self.radio_button_reenactment.isChecked():
            return
        if self.radio_button_simswap.isChecked():
            self.generate_simswap()
        elif self.radio_button_reenactment.isChecked():
            self.generate_face_reenactment()

    def change_tool_box(self):
        if self.tool_box.currentIndex() == 0:
            self.button_generate.setText("Generate")
            self.live_image_timer.start(30)
        elif self.tool_box.currentIndex() == 1:
            self.button_generate.setText("Try Again")
            self.live_image_timer.stop()

    def change_tabs(self):
        if self.tab_widget_input.currentIndex() == 0:
            self.live_image_timer.start(30)
        elif self.tab_widget_input.currentIndex() == 2:
            self.live_image_timer.stop()
            self.radio_button_simswap.setChecked(True)

    def change_combo_model(self):
        if self.combo_box_model.currentIndex() == 0:
            self.opt.name = "224"
        elif self.combo_box_model.currentIndex() == 1:
            self.opt.name = "230929"
        self.force_rebuild = True

    def replay_video(self):
        if self.deep_fake_video:
            self.deep_fake_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.deep_fake_video_timer.start(30)
    # endregion

    # region Live Images
    def update_webcam_image(self):
        ret, image = self.camera.read()
        if ret:
            resize_ratio = np.min([self.image_webcam.size().width()*0.95/image.shape[1],
                                   self.image_webcam.size().height()*0.95/image.shape[0]])
            image = cv2.resize(image, dsize=(int(image.shape[1]*resize_ratio), int(image.shape[0]*resize_ratio)))
            self.recorded_image = image.copy()

            cy = int(image.shape[0] * 0.5 - self.your_face_here.shape[0] * 0.5)
            cx = int(image.shape[1] * 0.5 - self.your_face_here.shape[1] * 0.5)
            y1, y2 = cy, cy + self.your_face_here.shape[0]
            x1, x2 = cx, cx + self.your_face_here.shape[1]

            alpha_s = self.your_face_here[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                image[y1:y2, x1:x2, c] = (alpha_s * self.your_face_here[:, :, c] +
                                          alpha_l * image[y1:y2, x1:x2, c])

            convert = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_BGR888)
            self.image_webcam.setPixmap(QPixmap.fromImage(convert))

            if self.record_mode:
                if not os.path.isdir(self.recording_path):
                    os.makedirs(self.recording_path)
                    self.video_timestamp_list.clear()

                img_to_save = cv2.resize(self.recorded_image, (self.recorded_image.shape[1]//2, self.recorded_image.shape[0]//2))
                data_name = "{:04d}.jpg".format(len(os.listdir(self.recording_path)))
                cv2.imwrite(os.path.join(self.recording_path, data_name), img_to_save)
                self.video_timestamp_list.append(time.time()*1000)

    def update_fake_video_image(self):
        if not self.deep_fake_video:
            self.deep_fake_video_timer.stop()
            return
        ret, image = self.deep_fake_video.read()
        if ret:
            resize_ratio = np.min([self.label_deep_fake_video.size().width()/image.shape[1],
                                   self.label_deep_fake_video.size().height()/image.shape[0]])
            image = cv2.resize(image, dsize=(int(image.shape[1]*resize_ratio), int(image.shape[0]*resize_ratio)))
            convert = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_BGR888)
            self.label_deep_fake_video.setPixmap(QPixmap.fromImage(convert))
        else:
            self.deep_fake_video_timer.stop()
    # endregion

    # region Misc Functions
    def set_output_images(self):
        image_size = np.clip(int(self.label_original_1.height()), 220, 500)

        # Original Images
        for i, label in enumerate([self.label_original_1, self.label_original_2]):
            image_path = r".\images\scenes\{:02d}\{:02d}.jpg".format(self.selected_clip, i+1)
            original_image = QPixmap(image_path).scaledToHeight(image_size)
            label.setPixmap(original_image)

        # Generated Images
        for i, label in enumerate([self.label_deep_fake_1, self.label_deep_fake_2]):
            image_path = r".\images\generated\{:02d}.jpg".format(i+1)
            original_image = QPixmap(image_path).scaledToHeight(image_size)
            label.setPixmap(original_image)

    def clear_output_images(self):
        self.label_original_1.clear()
        self.label_original_2.clear()
        self.label_deep_fake_1.clear()
        self.label_deep_fake_2.clear()

    def stop_recording(self):
        self.record_mode = False
        self.stop_recording_timer.stop()
        self.video_freq = np.mean([stamp - self.video_timestamp_list[i-1] for i, stamp in enumerate(self.video_timestamp_list) if i > 0])
        self.generate_face_reenactment(video_recorded=True)

        self.your_face_here = cv2.imread("./images/logos/your_face_here.png", cv2.IMREAD_UNCHANGED)
        self.your_face_here = cv2.resize(self.your_face_here, (0, 0), fx=self.marker_factor, fy=self.marker_factor)
    # endregion

    # region Generation Functions
    def generate_simswap(self):
        if self.tool_box.currentIndex() == 0:
            # 1. Deactivate tool box while operating
            self.tool_box.setEnabled(False)

            # 2. Detect & encode face in recorded image and target images
            self.encode_face_in_webcam_image()
            self.encode_target_face(r".\images\scenes\{:02d}\target_face.jpg".format(self.selected_clip))
            self.progress_bar.setValue(50)

            # 3. Apply Face Swap for three images of the selected scene
            for i, value in enumerate([75, 100]):
                final_image = self.face_swap_image(
                    r".\images\scenes\{:02d}\{:02d}.jpg".format(self.selected_clip, i + 1))
                final_image = cv2.resize(final_image, (512, 512))
                source_face_rescaled = cv2.resize(self.source_face, (128, 128))
                final_image[0:128, 0:128] = source_face_rescaled
                cv2.imwrite(r".\images\generated\{:02d}.jpg".format(i + 1), final_image)
                self.progress_bar.setValue(value)

            # 4. Switch to Output UI & show image results
            self.tool_box.setEnabled(True)
            self.tool_box_page_2.setEnabled(True)
            self.set_output_images()

            self.tool_box.setCurrentIndex(1)
            self.progress_bar.setValue(0)
            self.tool_box.repaint()

            # 5. Meanwhile, apply Face Swap to the whole video in background
            if not os.path.isfile(r".\videos\scenes\{:02d}\scene.mp4".format(self.selected_clip)):
                return

            # 1.5 If in Input Image mode, check if the video has been generated already
            if os.path.isfile("./videos/generated/pregenerated/{:02d}_{:02d}.mp4".format(self.selected_input,
                                                                                         self.selected_clip)):
                self.deep_fake_video = cv2.VideoCapture(r"./videos/generated/pregenerated/{:02d}_{:02d}.mp4".format(self.selected_input,
                                                                                         self.selected_clip))
                self.video_freq = 30

            elif not self.check_box_video.isChecked():
                return
            else:
                first_yield = True
                frame_count = 0
                for i in self.face_swap_video(r".\videos\scenes\{:02d}\scene.mp4".format(self.selected_clip)):
                    if first_yield:
                        frame_count = i
                        first_yield = False
                    if np.isnan(i):
                        break
                    self.tool_box.repaint()
                    self.progress_bar.setValue(int(i / frame_count * 100))

                # 6. Show Results
                self.deep_fake_video = cv2.VideoCapture(r".\videos\generated\scene.mp4")
            self.deep_fake_video_timer.start(self.video_freq)
            self.progress_bar.setValue(0)

        elif self.tool_box.currentIndex() == 1:
            self.tool_box.setCurrentIndex(0)

    def generate_face_reenactment(self, video_recorded=False):
        if self.tool_box.currentIndex() == 0:
            # 1. Start Recording
            if not video_recorded:
                if os.path.isdir(self.recording_path):
                    shutil.rmtree(self.recording_path)
                self.record_mode = True
                self.stop_recording_timer.start(6000)
                self.your_face_here = cv2.imread("./images/logos/your_face_here_recording.png", cv2.IMREAD_UNCHANGED)
                self.your_face_here = cv2.resize(self.your_face_here, (0, 0), fx=self.marker_factor, fy=self.marker_factor)

                return

            # 2. Deactivate tool box while operating
            self.tool_box.setEnabled(False)
            self.progress_bar.setValue(10)

            # 3. Extract face from recorded frames
            files = os.listdir(self.recording_path)
            files = [os.path.join(self.recording_path, f) for f in files]
            images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in files]

            imageio.mimsave("target.mp4", [img_as_ubyte(i) for i in images], fps=int(1000/self.video_freq))
            commands = process_video("target.mp4")
            self.progress_bar.setValue(25)
            subprocess.run(shlex.split(commands[0]))
            self.progress_bar.setValue(50)

            # 4. Create Face Reenactment
            self.create_reenactment()
            self.progress_bar.setValue(100)

            # 5. Switch to Output UI & show image results
            self.tool_box.setEnabled(True)
            self.tool_box_page_2.setEnabled(True)
            self.clear_output_images()

            self.tool_box.setCurrentIndex(1)
            self.progress_bar.setValue(0)
            self.tool_box.repaint()

            self.deep_fake_video = cv2.VideoCapture(r".\videos\generated\reenactment.mp4")
            self.deep_fake_video_timer.start(self.video_freq)
        elif self.tool_box.currentIndex() == 1:
            self.tool_box.setCurrentIndex(0)
    # endregion

    # region Reenactment Misc Functions
    def create_reenactment(self, find_best=True, relative=True, adapt_scale=False):
        source_image = imageio.v3.imread(r".\images\scenes\{:02d}\source.jpg".format(self.selected_clip))
        reader = imageio.get_reader("crop.mp4")
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()

        source_image = resize(source_image, (256, 256))[..., :3]
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
        generator, kp_detector = self.load_checkpoints(config_path="reenactment/config/vox-adv-256.yaml",
                                                       checkpoint_path="reenactment/models/DaGAN_vox_adv_256.pth.tar",
                                                       cpu=False)
        if find_best:
            i = find_best_frame(source_image, driving_video)
            driving_forward = driving_video[i:]
            driving_backward = driving_video[:(i + 1)][::-1]
            sources_forward, drivings_forward, predictions_forward, depth_forward = \
                self.make_animation(source_image,
                                    driving_forward,
                                    generator,
                                    kp_detector,
                                    relative=relative,
                                    adapt_movement_scale=adapt_scale,
                                    cpu=False)
            sources_backward, drivings_backward, predictions_backward, depth_backward = \
                self.make_animation(source_image,
                                    driving_backward,
                                    generator,
                                    kp_detector,
                                    relative=relative,
                                    adapt_movement_scale=adapt_scale,
                                    cpu=False)
            predictions = predictions_backward[::-1] + predictions_forward[1:]
            sources = sources_backward[::-1] + sources_forward[1:]
            drivings = drivings_backward[::-1] + drivings_forward[1:]
            depth_gray = depth_backward[::-1] + depth_forward[1:]
        else:
            sources, drivings, predictions, depth_gray = self.make_animation(source_image, driving_video, generator,
                                                                             kp_detector, relative=relative,
                                                                             adapt_movement_scale=adapt_scale,
                                                                             cpu=False)
        final_images = []
        for p, d, g in zip(predictions, drivings, depth_gray):
            g = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
            final_images.append(np.concatenate([d, g, p], axis=1))

        imageio.mimsave("videos/generated/reenactment.mp4", [img_as_ubyte(im) for im in final_images], fps=fps)

    def load_checkpoints(self, config_path, checkpoint_path, cpu=False):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config['model_params']['common_params']['num_kp'] = 15
        generator = getattr(GEN, "DepthAwareGenerator")(**config['model_params']['generator_params'],
                                                         **config['model_params']['common_params'])
        if not cpu:
            generator.cuda()
        config['model_params']['common_params']['num_channels'] = 4
        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                 **config['model_params']['common_params'])
        if not cpu:
            kp_detector.cuda()
        if cpu:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cuda:0")

        ckp_generator = OrderedDict((k.replace('module.', ''), v) for k, v in checkpoint['generator'].items())
        generator.load_state_dict(ckp_generator)
        ckp_kp_detector = OrderedDict((k.replace('module.', ''), v) for k, v in checkpoint['kp_detector'].items())
        kp_detector.load_state_dict(ckp_kp_detector)

        if not cpu:
            generator = DataParallelWithCallback(generator)
            kp_detector = DataParallelWithCallback(kp_detector)

        generator.eval()
        kp_detector.eval()

        return generator, kp_detector

    def make_animation(self, source_image, driving_video, generator, kp_detector,
                       relative=True, adapt_movement_scale=True,
                       cpu=False):
        sources = []
        drivings = []
        with torch.no_grad():
            predictions = []
            depth_gray = []
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
            if not cpu:
                source = source.cuda()
                driving = driving.cuda()
            outputs = self.depth_decoder(self.depth_encoder(source))
            depth_source = outputs[("disp", 0)]

            outputs = self.depth_decoder(self.depth_encoder(driving[:, :, 0]))
            depth_driving = outputs[("disp", 0)]
            source_kp = torch.cat((source, depth_source), 1)
            driving_kp = torch.cat((driving[:, :, 0], depth_driving), 1)

            kp_source = kp_detector(source_kp)
            kp_driving_initial = kp_detector(driving_kp)

            for frame_idx in tqdm(range(driving.shape[2])):
                driving_frame = driving[:, :, frame_idx]

                if not cpu:
                    driving_frame = driving_frame.cuda()
                outputs = self.depth_decoder(self.depth_encoder(driving_frame))
                depth_map = outputs[("disp", 0)]

                gray_driving = np.transpose(depth_map.data.cpu().numpy(), [0, 2, 3, 1])[0]
                gray_driving = 1 - gray_driving / np.max(gray_driving)

                frame = torch.cat((driving_frame, depth_map), 1)
                kp_driving = kp_detector(frame)

                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                       kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                       use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
                out = generator(source, kp_source=kp_source, kp_driving=kp_norm, source_depth=depth_source,
                                driving_depth=depth_map)

                drivings.append(np.transpose(driving_frame.data.cpu().numpy(), [0, 2, 3, 1])[0])
                sources.append(np.transpose(source.data.cpu().numpy(), [0, 2, 3, 1])[0])
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
                depth_gray.append(gray_driving)
        return sources, drivings, predictions, depth_gray

    # endregion

    # region Simswap Misc Functions
    def encode_face_in_webcam_image(self):
        if self.tab_widget_input.currentIndex() == 1:
            self.recorded_image = cv2.imread(r".\images\input\{:02d}.jpg".format(self.selected_input))
        # Detect face in webcam image and extract features
        with torch.no_grad():
            source_face_image, _, bboxes = self.face_det_model.get(self.recorded_image, self.opt.crop_size)
            bbox_sizes = [b[2]*b[3] for b in bboxes]
            max_size_idx = np.argmax(bbox_sizes)

            self.source_face = source_face_image[max_size_idx].copy()
            source_face_image = cv2.cvtColor(source_face_image[max_size_idx], cv2.COLOR_BGR2RGB)
            source_face_image = Image.fromarray(source_face_image)
            source_face_image = transformer_Arcface(source_face_image)
            source_face = source_face_image.view(-1, source_face_image.shape[0],
                                                 source_face_image.shape[1],
                                                 source_face_image.shape[2])
            # Convert numpy to tensor
            source_face = source_face.cuda()

            # Create latent id
            source_image_downsample = F.interpolate(source_face, size=(112, 112), mode='bicubic')
            latent_source_id = self.face_swap_model.netArc(source_image_downsample)
            self.source_id = F.normalize(latent_source_id, p=2, dim=1)

    def encode_target_face(self, target_path):
        # Detect the specific person to be swapped in the provided image
        with torch.no_grad():
            target_face_whole = cv2.imread(target_path)
            target_face_align_crop, _, _ = self.face_det_model.get(target_face_whole, self.opt.crop_size)
            target_face_align_crop = cv2.cvtColor(target_face_align_crop[0], cv2.COLOR_BGR2RGB)
            target_face_align_crop = Image.fromarray(target_face_align_crop)
            target_face = transformer_Arcface(target_face_align_crop)
            target_face = target_face.view(-1, target_face.shape[0], target_face.shape[1], target_face.shape[2])
            self.target_face = target_face.clone().detach().numpy()[0].transpose(1, 2, 0)

            target_face = target_face.cuda()
            target_face_downsample = F.interpolate(target_face, size=(112, 112), mode='bicubic')

            self.target_id = self.face_swap_model.netArc(target_face_downsample)
            # self.target_id = F.normalize(self.target_id, p=2, dim=1)

    def face_swap_image(self, target_path):
        target_image = cv2.imread(target_path)
        detection_result = self.face_det_model.get(target_image, self.opt.crop_size)

        if not detection_result:
            return target_image

        face_image_list, image_mat_list = detection_result[0], detection_result[1]

        id_errors = []
        image_tensor_list = []
        image_tensor_list_transformed = []
        for face_image in face_image_list:
            face_image_tensor = _to_tensor(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))[None, ...].cuda()
            face_image_tensor_norm = spNorm(face_image_tensor)
            face_image_tensor_norm_ds = F.interpolate(face_image_tensor_norm, size=(112, 112), mode='bicubic')
            face_id = self.face_swap_model.netArc(face_image_tensor_norm_ds)

            id_errors.append(mse(face_id, self.target_id).detach().cpu().numpy())
            image_tensor_list.append(face_image_tensor)
            image_tensor_list_transformed.append(face_image_tensor_norm)

        id_errors_array = np.array(id_errors)
        min_index = np.argmin(id_errors_array)
        min_value = id_errors_array[min_index]

        if min_value < self.opt.id_thres or np.isnan(min_value):
            if self.opt.name == "224":
                swap_result = self.face_swap_model(None, image_tensor_list[min_index],
                                                   self.source_id, None, True)[0]
            else:
                swap_result = self.face_swap_model(None, image_tensor_list_transformed[min_index],
                                                   self.source_id, None, True)[0]

            final_image = self.reverse_2_whole_image(image_tensor_list[min_index], swap_result,
                                                     image_mat_list[min_index],
                                                     self.opt.crop_size, target_image, parsing_model=self.mask_net,
                                                     use_mask=self.opt.use_mask, norm=spNorm)
            return final_image

    def face_swap_video(self, target_path, temp_results_dir="./temp"):
        # Open Video File
        video_audio_clip = AudioFileClip(target_path)
        video = cv2.VideoCapture(target_path)

        # Get Video Information
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        self.video_freq = 30
        yield frame_count

        # Delete temporary folder and recreate
        if os.path.exists(temp_results_dir):
            shutil.rmtree(temp_results_dir)
        if not os.path.exists(temp_results_dir):
            os.mkdir(temp_results_dir)

        # Iterate through the video
        for frame_index in range(frame_count):
            ret, frame = video.read()
            if ret:
                # Detect faces in the image
                detection_result = self.face_det_model.get(frame, self.opt.crop_size)

                if detection_result is not None:
                    face_image_list, image_mat_list = detection_result[0], detection_result[1]

                    id_errors = []
                    image_tensor_list = []
                    for face_image in face_image_list:
                        face_image_tensor = _to_tensor(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))[None, ...].cuda()

                        face_image_tensor_norm = spNorm(face_image_tensor)
                        face_image_tensor_norm = F.interpolate(face_image_tensor_norm, size=(112, 112), mode='bicubic')
                        face_id = self.face_swap_model.netArc(face_image_tensor_norm)

                        id_errors.append(mse(face_id, self.target_id).detach().cpu().numpy())
                        image_tensor_list.append(face_image_tensor)

                    id_errors_array = np.array(id_errors)
                    min_index = np.argmin(id_errors_array)
                    min_value = id_errors_array[min_index]

                    if min_value < self.opt.id_thres or np.isnan(min_value):
                        swap_result = \
                            self.face_swap_model(None, image_tensor_list[min_index], self.source_id, None, True)[0]

                        final_image = self.reverse_2_whole_image(image_tensor_list[min_index], swap_result,
                                                                 image_mat_list[min_index],
                                                                 self.opt.crop_size, frame,
                                                                 parsing_model=self.mask_net,
                                                                 use_mask=self.opt.use_mask, norm=spNorm)
                    else:
                        final_image = frame.astype(np.uint8)
                else:
                    final_image = frame.astype(np.uint8)
                source_face_rescaled = cv2.resize(self.source_face, (256, 256))
                final_image[0:256, 0:256] = source_face_rescaled
                final_image = np.concatenate([frame, final_image], axis=0)
                cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:07d}.jpg'.format(frame_index)), final_image)
                yield frame_index
            else:
                break

        video.release()
        path = os.path.join(temp_results_dir, '*.jpg')
        image_filenames = sorted(glob.glob(path))

        video_clip = ImageSequenceClip(image_filenames, fps=fps)
        video_clip = video_clip.set_audio(video_audio_clip)
        video_clip.write_videofile("./videos/generated/scene.mp4", audio_codec='aac')
        if self.tab_widget_input.currentIndex() == 1:
            video_clip.write_videofile("./videos/generated/pregenerated/{:02d}_{:02d}.mp4".format(self.selected_input,
                                                                                                  self.selected_clip),
                                       audio_codec='aac')
        # video_audio_clip = AudioFileClip(target_path)
        yield np.nan

    def reverse_2_whole_image(self, image_tensor, swapped_image, image_mat, crop_size, original_image,
                              parsing_model=None, norm=None, use_mask=False):

        if use_mask:
            smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=7).cuda()
        else:
            smooth_mask = None

        swapped_img = swapped_image.cpu().detach().numpy().transpose((1, 2, 0))
        img_white = np.full((crop_size, crop_size), 255, dtype=float)

        # inverse the Affine transformation matrix
        # mat_rev_alt = np.linalg.inv(image_mat)
        mat_rev = np.zeros([2, 3])
        div1 = image_mat[0][0] * image_mat[1][1] - image_mat[0][1] * image_mat[1][0]
        mat_rev[0][0] = image_mat[1][1] / div1
        mat_rev[0][1] = -image_mat[0][1] / div1
        mat_rev[0][2] = -(image_mat[0][2] * image_mat[1][1] - image_mat[0][1] * image_mat[1][2]) / div1
        div2 = image_mat[0][1] * image_mat[1][0] - image_mat[0][0] * image_mat[1][1]
        mat_rev[1][0] = image_mat[1][0] / div2
        mat_rev[1][1] = -image_mat[0][0] / div2
        mat_rev[1][2] = -(image_mat[0][2] * image_mat[1][0] - image_mat[0][0] * image_mat[1][2]) / div2

        original_size = (original_image.shape[1], original_image.shape[0])
        if use_mask:
            source_img_norm = norm(image_tensor)
            source_img_512 = F.interpolate(source_img_norm, size=(512, 512))
            out = parsing_model(source_img_512)[0]
            parsing = out.squeeze(0).detach().cpu().numpy().argmax(0)
            vis_parsing_anno = parsing.copy().astype(np.uint8)
            target_mask = encode_segmentation_rgb(vis_parsing_anno, no_neck=False)
            if target_mask.sum() >= 5000:
                target_mask = cv2.resize(target_mask, (crop_size, crop_size))
                target_image_parsing = self.postprocess(swapped_img,
                                                        image_tensor[0].cpu().detach().numpy().transpose((1, 2, 0)),
                                                        target_mask, smooth_mask)
                target_image = cv2.warpAffine(target_image_parsing, mat_rev, original_size)
            else:
                target_image = cv2.warpAffine(swapped_img, mat_rev, original_size)[..., ::-1]
        else:
            # ToDo: Fix
            target_image = cv2.warpAffine(swapped_img.astype(np.uint8), mat_rev, original_size)

        img_white = cv2.warpAffine(img_white, mat_rev, original_size)
        img_white[img_white > 20] = 255
        img_mask = img_white

        kernel = np.ones((40, 40), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        kernel_size = (20, 20)
        blur_size = tuple(2 * i + 1 for i in kernel_size)
        img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
        img_mask /= 255

        img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])

        if use_mask:
            target_image = np.array(target_image, dtype=np.float32) * 255
        else:
            target_image = np.array(target_image, dtype=np.float32)[..., ::-1] * 255

        img = np.array(original_image, dtype=np.float32)
        img = img_mask * target_image + (1 - img_mask) * img

        final_img = img.astype(np.uint8)
        return final_img

    def postprocess(self, swapped_face, target, target_mask, smooth_mask):
        mask_tensor = torch.from_numpy(target_mask.copy().transpose((2, 0, 1))).float().mul_(1 / 255.0).cuda()
        face_mask_tensor = mask_tensor[0] + mask_tensor[1]

        soft_face_mask_tensor, _ = smooth_mask(face_mask_tensor.unsqueeze_(0).unsqueeze_(0))
        soft_face_mask_tensor.squeeze_()

        soft_face_mask = soft_face_mask_tensor.cpu().numpy()
        soft_face_mask = soft_face_mask[:, :, np.newaxis]

        result = swapped_face * soft_face_mask + target * (1 - soft_face_mask)
        result = result[:, :, ::-1]
        return result
    # endregion


def find_best_frame(source, driving):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=True,
                                      device='cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


def encode_segmentation_rgb(segmentation, no_neck=True):
    parse = segmentation

    face_part_ids = [1, 2, 3, 4, 5, 6, 10, 12, 13] if no_neck else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14]
    mouth_id = 11
    # hair_id = 17
    face_map = np.zeros([parse.shape[0], parse.shape[1]])
    mouth_map = np.zeros([parse.shape[0], parse.shape[1]])
    # hair_map = np.zeros([parse.shape[0], parse.shape[1]])

    for valid_id in face_part_ids:
        valid_index = np.where(parse == valid_id)
        face_map[valid_index] = 255
    valid_index = np.where(parse == mouth_id)
    mouth_map[valid_index] = 255
    return np.stack([face_map, mouth_map], axis=2)


class SoftErosion(torch.nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()
        return x, mask


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == '__main__':
    main()
