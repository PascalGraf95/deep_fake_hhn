# Form implementation generated from reading ui file 'deepfakehhn.ui'
#
# Created by: PyQt6 UI code generator 6.5.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_DeepFakeHHN(object):
    def setupUi(self, DeepFakeHHN):
        DeepFakeHHN.setObjectName("DeepFakeHHN")
        DeepFakeHHN.resize(1938, 1080)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(DeepFakeHHN.sizePolicy().hasHeightForWidth())
        DeepFakeHHN.setSizePolicy(sizePolicy)
        DeepFakeHHN.setMinimumSize(QtCore.QSize(1920, 1080))
        DeepFakeHHN.setMaximumSize(QtCore.QSize(2560, 1440))
        DeepFakeHHN.setAutoFillBackground(False)
        self.centralwidget = QtWidgets.QWidget(parent=DeepFakeHHN)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetDefaultConstraint)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_title = QtWidgets.QLabel(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_title.sizePolicy().hasHeightForWidth())
        self.label_title.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiBold")
        font.setPointSize(26)
        font.setBold(False)
        self.label_title.setFont(font)
        self.label_title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_title.setObjectName("label_title")
        self.horizontalLayout_2.addWidget(self.label_title)
        self.image_saai_logo = QtWidgets.QLabel(parent=self.centralwidget)
        self.image_saai_logo.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_saai_logo.setObjectName("image_saai_logo")
        self.horizontalLayout_2.addWidget(self.image_saai_logo)
        self.image_hhn_logo = QtWidgets.QLabel(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image_hhn_logo.sizePolicy().hasHeightForWidth())
        self.image_hhn_logo.setSizePolicy(sizePolicy)
        self.image_hhn_logo.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_hhn_logo.setObjectName("image_hhn_logo")
        self.horizontalLayout_2.addWidget(self.image_hhn_logo)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.label_subtitle = QtWidgets.QLabel(parent=self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setItalic(True)
        self.label_subtitle.setFont(font)
        self.label_subtitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_subtitle.setObjectName("label_subtitle")
        self.verticalLayout.addWidget(self.label_subtitle)
        self.tool_box = QtWidgets.QToolBox(parent=self.centralwidget)
        self.tool_box.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tool_box.sizePolicy().hasHeightForWidth())
        self.tool_box.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.tool_box.setFont(font)
        self.tool_box.setObjectName("tool_box")
        self.tool_box_page_1 = QtWidgets.QWidget()
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(False)
        self.tool_box_page_1.setFont(font)
        self.tool_box_page_1.setAccessibleName("")
        self.tool_box_page_1.setObjectName("tool_box_page_1")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tool_box_page_1)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_live_image = QtWidgets.QLabel(parent=self.tool_box_page_1)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.label_live_image.setFont(font)
        self.label_live_image.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_live_image.setObjectName("label_live_image")
        self.verticalLayout_2.addWidget(self.label_live_image)
        self.image_webcam = QtWidgets.QLabel(parent=self.tool_box_page_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image_webcam.sizePolicy().hasHeightForWidth())
        self.image_webcam.setSizePolicy(sizePolicy)
        self.image_webcam.setMinimumSize(QtCore.QSize(0, 0))
        self.image_webcam.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_webcam.setObjectName("image_webcam")
        self.verticalLayout_2.addWidget(self.image_webcam)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.image_preview_1 = QtWidgets.QLabel(parent=self.tool_box_page_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.image_preview_1.sizePolicy().hasHeightForWidth())
        self.image_preview_1.setSizePolicy(sizePolicy)
        self.image_preview_1.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.image_preview_1.setFrameShadow(QtWidgets.QFrame.Shadow.Plain)
        self.image_preview_1.setLineWidth(1)
        self.image_preview_1.setMidLineWidth(0)
        self.image_preview_1.setScaledContents(False)
        self.image_preview_1.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_preview_1.setObjectName("image_preview_1")
        self.verticalLayout_3.addWidget(self.image_preview_1)
        self.button_preview_1 = QtWidgets.QPushButton(parent=self.tool_box_page_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_preview_1.sizePolicy().hasHeightForWidth())
        self.button_preview_1.setSizePolicy(sizePolicy)
        self.button_preview_1.setMaximumSize(QtCore.QSize(16777214, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        self.button_preview_1.setFont(font)
        self.button_preview_1.setObjectName("button_preview_1")
        self.verticalLayout_3.addWidget(self.button_preview_1)
        self.gridLayout.addLayout(self.verticalLayout_3, 0, 0, 1, 1)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.image_preview_2 = QtWidgets.QLabel(parent=self.tool_box_page_1)
        self.image_preview_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_preview_2.setObjectName("image_preview_2")
        self.verticalLayout_4.addWidget(self.image_preview_2)
        self.button_preview_2 = QtWidgets.QPushButton(parent=self.tool_box_page_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_preview_2.sizePolicy().hasHeightForWidth())
        self.button_preview_2.setSizePolicy(sizePolicy)
        self.button_preview_2.setMaximumSize(QtCore.QSize(16777214, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        self.button_preview_2.setFont(font)
        self.button_preview_2.setObjectName("button_preview_2")
        self.verticalLayout_4.addWidget(self.button_preview_2)
        self.gridLayout.addLayout(self.verticalLayout_4, 0, 1, 1, 1)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.image_preview_3 = QtWidgets.QLabel(parent=self.tool_box_page_1)
        self.image_preview_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_preview_3.setObjectName("image_preview_3")
        self.verticalLayout_5.addWidget(self.image_preview_3)
        self.button_preview_3 = QtWidgets.QPushButton(parent=self.tool_box_page_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_preview_3.sizePolicy().hasHeightForWidth())
        self.button_preview_3.setSizePolicy(sizePolicy)
        self.button_preview_3.setMaximumSize(QtCore.QSize(16777214, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        self.button_preview_3.setFont(font)
        self.button_preview_3.setObjectName("button_preview_3")
        self.verticalLayout_5.addWidget(self.button_preview_3)
        self.gridLayout.addLayout(self.verticalLayout_5, 1, 0, 1, 1)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.image_preview_4 = QtWidgets.QLabel(parent=self.tool_box_page_1)
        self.image_preview_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_preview_4.setObjectName("image_preview_4")
        self.verticalLayout_6.addWidget(self.image_preview_4)
        self.button_preview_4 = QtWidgets.QPushButton(parent=self.tool_box_page_1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum, QtWidgets.QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_preview_4.sizePolicy().hasHeightForWidth())
        self.button_preview_4.setSizePolicy(sizePolicy)
        self.button_preview_4.setMaximumSize(QtCore.QSize(16777214, 16777215))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(False)
        self.button_preview_4.setFont(font)
        self.button_preview_4.setObjectName("button_preview_4")
        self.verticalLayout_6.addWidget(self.button_preview_4)
        self.gridLayout.addLayout(self.verticalLayout_6, 1, 1, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)
        self.gridLayout_4.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        self.tool_box.addItem(self.tool_box_page_1, "")
        self.tool_box_page_2 = QtWidgets.QWidget()
        self.tool_box_page_2.setEnabled(False)
        self.tool_box_page_2.setObjectName("tool_box_page_2")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tool_box_page_2)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(parent=self.tool_box_page_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_deep_fake_3 = QtWidgets.QLabel(parent=self.tool_box_page_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_deep_fake_3.sizePolicy().hasHeightForWidth())
        self.label_deep_fake_3.setSizePolicy(sizePolicy)
        self.label_deep_fake_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_deep_fake_3.setObjectName("label_deep_fake_3")
        self.gridLayout_3.addWidget(self.label_deep_fake_3, 3, 1, 1, 1)
        self.label_deep_fake_1 = QtWidgets.QLabel(parent=self.tool_box_page_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_deep_fake_1.sizePolicy().hasHeightForWidth())
        self.label_deep_fake_1.setSizePolicy(sizePolicy)
        self.label_deep_fake_1.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_deep_fake_1.setObjectName("label_deep_fake_1")
        self.gridLayout_3.addWidget(self.label_deep_fake_1, 1, 1, 1, 1)
        self.label_original_3 = QtWidgets.QLabel(parent=self.tool_box_page_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_original_3.sizePolicy().hasHeightForWidth())
        self.label_original_3.setSizePolicy(sizePolicy)
        self.label_original_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_original_3.setObjectName("label_original_3")
        self.gridLayout_3.addWidget(self.label_original_3, 3, 0, 1, 1)
        self.label_original_2 = QtWidgets.QLabel(parent=self.tool_box_page_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_original_2.sizePolicy().hasHeightForWidth())
        self.label_original_2.setSizePolicy(sizePolicy)
        self.label_original_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_original_2.setObjectName("label_original_2")
        self.gridLayout_3.addWidget(self.label_original_2, 2, 0, 1, 1)
        self.label_deep_fake_2 = QtWidgets.QLabel(parent=self.tool_box_page_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_deep_fake_2.sizePolicy().hasHeightForWidth())
        self.label_deep_fake_2.setSizePolicy(sizePolicy)
        self.label_deep_fake_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_deep_fake_2.setObjectName("label_deep_fake_2")
        self.gridLayout_3.addWidget(self.label_deep_fake_2, 2, 1, 1, 1)
        self.label_original_1 = QtWidgets.QLabel(parent=self.tool_box_page_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_original_1.sizePolicy().hasHeightForWidth())
        self.label_original_1.setSizePolicy(sizePolicy)
        self.label_original_1.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_original_1.setObjectName("label_original_1")
        self.gridLayout_3.addWidget(self.label_original_1, 1, 0, 1, 1)
        self.label_original = QtWidgets.QLabel(parent=self.tool_box_page_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_original.sizePolicy().hasHeightForWidth())
        self.label_original.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.label_original.setFont(font)
        self.label_original.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_original.setObjectName("label_original")
        self.gridLayout_3.addWidget(self.label_original, 0, 0, 1, 1)
        self.label_deep_fake = QtWidgets.QLabel(parent=self.tool_box_page_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_deep_fake.sizePolicy().hasHeightForWidth())
        self.label_deep_fake.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        self.label_deep_fake.setFont(font)
        self.label_deep_fake.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_deep_fake.setObjectName("label_deep_fake")
        self.gridLayout_3.addWidget(self.label_deep_fake, 0, 1, 1, 1)
        self.horizontalLayout_3.addLayout(self.gridLayout_3)
        self.gridLayout_5.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
        self.tool_box.addItem(self.tool_box_page_2, "")
        self.verticalLayout.addWidget(self.tool_box)
        self.button_generate = QtWidgets.QPushButton(parent=self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Bahnschrift SemiBold")
        font.setPointSize(18)
        self.button_generate.setFont(font)
        self.button_generate.setObjectName("button_generate")
        self.verticalLayout.addWidget(self.button_generate)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.progress_bar = QtWidgets.QProgressBar(parent=self.centralwidget)
        self.progress_bar.setEnabled(True)
        self.progress_bar.setProperty("value", 0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setObjectName("progress_bar")
        self.gridLayout_2.addWidget(self.progress_bar, 1, 0, 1, 1)
        DeepFakeHHN.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=DeepFakeHHN)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1938, 22))
        self.menubar.setObjectName("menubar")
        DeepFakeHHN.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=DeepFakeHHN)
        self.statusbar.setObjectName("statusbar")
        DeepFakeHHN.setStatusBar(self.statusbar)

        self.retranslateUi(DeepFakeHHN)
        QtCore.QMetaObject.connectSlotsByName(DeepFakeHHN)

    def retranslateUi(self, DeepFakeHHN):
        _translate = QtCore.QCoreApplication.translate
        DeepFakeHHN.setWindowTitle(_translate("DeepFakeHHN", "DeepFakeHHN"))
        self.label_title.setText(_translate("DeepFakeHHN", "HHN Deep Fake Generator"))
        self.image_saai_logo.setText(_translate("DeepFakeHHN", "SAAI Logo"))
        self.image_hhn_logo.setText(_translate("DeepFakeHHN", "HHN LOGO"))
        self.label_subtitle.setText(_translate("DeepFakeHHN", "Take the central role in any movie!"))
        self.label_live_image.setText(_translate("DeepFakeHHN", "Live Image"))
        self.image_webcam.setText(_translate("DeepFakeHHN", "Webcam Image"))
        self.image_preview_1.setText(_translate("DeepFakeHHN", "Image Preview 1"))
        self.button_preview_1.setText(_translate("DeepFakeHHN", "Select Video"))
        self.image_preview_2.setText(_translate("DeepFakeHHN", "Image Preview 2"))
        self.button_preview_2.setText(_translate("DeepFakeHHN", "Select Video"))
        self.image_preview_3.setText(_translate("DeepFakeHHN", "Image Preview 3"))
        self.button_preview_3.setText(_translate("DeepFakeHHN", "Select Video"))
        self.image_preview_4.setText(_translate("DeepFakeHHN", "Image Preview 4"))
        self.button_preview_4.setText(_translate("DeepFakeHHN", "Select Video"))
        self.tool_box.setItemText(self.tool_box.indexOf(self.tool_box_page_1), _translate("DeepFakeHHN", "Recording"))
        self.label_2.setText(_translate("DeepFakeHHN", "TextLabel"))
        self.label_deep_fake_3.setText(_translate("DeepFakeHHN", "TextLabel"))
        self.label_deep_fake_1.setText(_translate("DeepFakeHHN", "TextLabel"))
        self.label_original_3.setText(_translate("DeepFakeHHN", "TextLabel"))
        self.label_original_2.setText(_translate("DeepFakeHHN", "TextLabel"))
        self.label_deep_fake_2.setText(_translate("DeepFakeHHN", "TextLabel"))
        self.label_original_1.setText(_translate("DeepFakeHHN", "TextLabel"))
        self.label_original.setText(_translate("DeepFakeHHN", "Original"))
        self.label_deep_fake.setText(_translate("DeepFakeHHN", "Deep Fake"))
        self.tool_box.setItemText(self.tool_box.indexOf(self.tool_box_page_2), _translate("DeepFakeHHN", "Ouput"))
        self.button_generate.setText(_translate("DeepFakeHHN", "Generate"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    DeepFakeHHN = QtWidgets.QMainWindow()
    ui = Ui_DeepFakeHHN()
    ui.setupUi(DeepFakeHHN)
    DeepFakeHHN.show()
    sys.exit(app.exec())
