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
    id_thres = 0.04
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

        # region Bug Fix
        self.tool_box.setCurrentIndex(1)
        time.sleep(0.5)
        self.tool_box.setCurrentIndex(0)

        # region - Internal Variables -
        # Simswap Objects
        self.opt = None
        self.face_swap_model = None
        self.face_det_model = None
        self.mask_net = None
        # Encoded Faces & Images
        self.source_face = None
        self.source_id = None
        self.target_face = None
        self.target_id = None
        # region State Variables and Objects
        self.selected_clip = 0
        self.recorded_image = None
        self.movie = QMovie("./loader.gif")
        self.video_fps = 30
        self.deep_fake_video = None
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

        # Setup Deep Fake Result Video
        self.deep_fake_video_timer = QTimer()
        self.deep_fake_video_timer.timeout.connect(self.update_fake_video_image)


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
            # Deactivate tool box while operating
            self.tool_box.setEnabled(False)

            # 1. Initialize Simswap Models
            self.initialize_models()
            self.progress_bar.setValue(25)

            # 2. Detect & encode face in recorded image and target images
            self.encode_face_in_webcam_image()
            self.encode_target_face(r".\images\scenes\{:02d}\target_face.jpg".format(self.selected_clip))
            self.progress_bar.setValue(50)

            # 3. Apply Face Swap for three images of the selected scene
            for i, value in enumerate([75, 100]):
                final_image = self.face_swap_image(r".\images\scenes\{:02d}\{:02d}.jpg".format(self.selected_clip, i+1))
                cv2.imwrite(r".\images\generated\{:02d}.jpg".format(i+1), final_image)
                self.progress_bar.setValue(value)

            # 4. Switch to Output UI & show image results
            self.tool_box.setEnabled(True)
            self.tool_box_page_2.setEnabled(True)
            self.set_output_images()

            self.tool_box.setCurrentIndex(1)
            self.progress_bar.setValue(0)
            self.tool_box.repaint()

            # 5. Meanwhile, apply Face Swap to the whole video in background
            first_yield = True
            frame_count = 0
            for i in self.face_swap_video(r".\videos\scenes\{:02d}\scene.mp4".format(self.selected_clip)):
                if first_yield:
                    frame_count = i
                    first_yield = False
                if np.isnan(i):
                    break
                self.tool_box.repaint()
                self.progress_bar.setValue(int(i/frame_count*100))
                
            # 6. Show Results
            self.deep_fake_video = cv2.VideoCapture(r".\videos\generated\scene.mp4")
            self.deep_fake_video_timer.start(30)
            self.progress_bar.setValue(0)

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
        for i, label in enumerate([self.label_original_1, self.label_original_2]):
            image_path = r".\images\scenes\{:02d}\{:02d}.jpg".format(self.selected_clip, i+1)
            original_image = QPixmap(image_path).scaledToHeight(image_size)
            label.setPixmap(original_image)

        # Generated Images
        for i, label in enumerate([self.label_deep_fake_1, self.label_deep_fake_2]):
            image_path = r".\images\generated\{:02d}.jpg".format(i+1)
            original_image = QPixmap(image_path).scaledToHeight(image_size)
            label.setPixmap(original_image)
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

    def update_fake_video_image(self):
        ret, image = self.deep_fake_video.read()
        if ret:
            resize_ratio = np.min([self.label_2.size().width()/image.shape[1],
                                   self.label_2.size().height()/image.shape[0]])
            image = cv2.resize(image, dsize=(int(image.shape[1]*resize_ratio), int(image.shape[0]*resize_ratio)))
            self.recorded_image = image
            convert = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_BGR888)
            self.label_2.setPixmap(QPixmap.fromImage(convert))

    def update_background_image(self):
        ret, image = self.background_clip.read()
        cv2.imwrite("./images/background_image.png", image)
        stylesheet = 'background-image: url("./images/background_image.png");'
        self.centralwidget.setStyleSheet(stylesheet)

    # endregion

    # region Simswap Functions
    def initialize_models(self):
        if not self.opt:
            torch.nn.Module.dump_patches = True
            self.opt = FaceDetectionOptions()
        if not self.face_swap_model:
            self.face_swap_model = create_model(self.opt)
            self.face_swap_model.eval()
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

    def encode_face_in_webcam_image(self):
        # Detect face in webcam image and extract features
        with torch.no_grad():
            source_face_image, _ = self.face_det_model.get(self.recorded_image, self.opt.crop_size)
            source_face_image = cv2.cvtColor(source_face_image[0], cv2.COLOR_BGR2RGB)
            source_face_image_pil = Image.fromarray(source_face_image)
            source_face_image = transformer_Arcface(source_face_image_pil)
            source_face = source_face_image.view(-1, source_face_image.shape[0],
                                                 source_face_image.shape[1],
                                                 source_face_image.shape[2])
            self.source_face = source_face.clone().detach().numpy()[0].transpose(1, 2, 0)

            # Convert numpy to tensor
            source_face = source_face.cuda()

            # Create latent id
            source_image_downsample = F.interpolate(source_face, size=(112, 112))
            latent_source_id = self.face_swap_model.netArc(source_image_downsample)
            self.source_id = F.normalize(latent_source_id, p=2, dim=1)

            # cv2.imshow("SOURCE FACE", cv2.cvtColor(source_image_downsample.cpu().detach().numpy()[0].transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)

    def encode_target_face(self, target_path):
        # Detect the specific person to be swapped in the provided image
        with torch.no_grad():
            target_face_whole = cv2.imread(target_path)
            target_face_align_crop, _ = self.face_det_model.get(target_face_whole, self.opt.crop_size)
            target_face_align_crop = cv2.cvtColor(target_face_align_crop[0], cv2.COLOR_BGR2RGB)
            target_face_align_crop_pil = Image.fromarray(target_face_align_crop)
            target_face = transformer_Arcface(target_face_align_crop_pil)
            target_face = target_face.view(-1, target_face.shape[0], target_face.shape[1], target_face.shape[2])
            self.target_face = target_face.clone().detach().numpy()[0].transpose(1, 2, 0)

            target_face = target_face.cuda()
            target_face_downsample = F.interpolate(target_face, size=(112, 112))

            # cv2.imshow("TARGET FACE", cv2.cvtColor(target_face_downsample.cpu().detach().numpy()[0].transpose(1, 2, 0), cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)
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
        for face_image in face_image_list:
            face_image_tensor = _to_tensor(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))[None, ...].cuda()

            face_image_tensor_norm = spNorm(face_image_tensor)
            face_image_tensor_norm = F.interpolate(face_image_tensor_norm, size=(112, 112))
            face_id = self.face_swap_model.netArc(face_image_tensor_norm)

            id_errors.append(mse(face_id, self.target_id).detach().cpu().numpy())
            image_tensor_list.append(face_image_tensor)

        id_errors_array = np.array(id_errors)
        min_index = np.argmin(id_errors_array)
        min_value = id_errors_array[min_index]

        if min_value < self.opt.id_thres or np.isnan(min_value):
            swap_result = self.face_swap_model(None, image_tensor_list[min_index], self.source_id, None, True)[0]

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
        self.video_fps = fps
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
                        face_image_tensor_norm = F.interpolate(face_image_tensor_norm, size=(112, 112))
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

                cv2.imwrite(os.path.join(temp_results_dir, 'frame_{:07d}.jpg'.format(frame_index)), final_image)
                print(frame_index)
                yield frame_index
            else:
                break

        video.release()
        path = os.path.join(temp_results_dir, '*.jpg')
        image_filenames = sorted(glob.glob(path))

        video_clip = ImageSequenceClip(image_filenames, fps=fps)
        video_clip = video_clip.set_audio(video_audio_clip)
        video_clip.write_videofile("./videos/generated/scene.mp4", audio_codec='aac')
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
                target_image = cv2.warpAffine(swapped_image, mat_rev, original_size)[..., ::-1]
        else:
            # ToDo: Fix
            target_image = cv2.warpAffine(swapped_image.astype(np.uint8), mat_rev, original_size)

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

    """
    def face_swap_video(self):
        # Given the source and target person ids, swap faces in the provided video
        with torch.no_grad():
            video_swap(opt.video_path, latent_source_id, target_face_id_nonorm, opt.id_thres,
                       model, app, opt.output_path, temp_results_dir=opt.temp_path, no_simswaplogo=True,
                       use_mask=opt.use_mask, crop_size=crop_size)
    """


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
