# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'video.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_video(object):
    def setupUi(self, video):
        video.setObjectName("video")
        video.resize(900, 600)
        self.upload = QtWidgets.QPushButton(video)
        self.upload.setGeometry(QtCore.QRect(660, 540, 93, 28))
        self.upload.setObjectName("upload")
        self.back = QtWidgets.QPushButton(video)
        self.back.setGeometry(QtCore.QRect(770, 540, 93, 28))
        self.back.setObjectName("back")
        self.scrollArea = QtWidgets.QScrollArea(video)
        self.scrollArea.setGeometry(QtCore.QRect(60, 30, 771, 441))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 769, 439))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.videoframe = QtWidgets.QLabel(self.scrollAreaWidgetContents)
        self.videoframe.setGeometry(QtCore.QRect(30, 20, 72, 15))
        self.videoframe.setText("")
        self.videoframe.setObjectName("videoframe")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.scrollArea_2 = QtWidgets.QScrollArea(video)
        self.scrollArea_2.setGeometry(QtCore.QRect(60, 500, 431, 71))
        self.scrollArea_2.setWidgetResizable(True)
        self.scrollArea_2.setObjectName("scrollArea_2")
        self.scrollAreaWidgetContents_2 = QtWidgets.QWidget()
        self.scrollAreaWidgetContents_2.setGeometry(QtCore.QRect(0, 0, 429, 69))
        self.scrollAreaWidgetContents_2.setObjectName("scrollAreaWidgetContents_2")
        self.resultlable = QtWidgets.QLabel(self.scrollAreaWidgetContents_2)
        self.resultlable.setGeometry(QtCore.QRect(20, 10, 391, 51))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(16)
        self.resultlable.setFont(font)
        self.resultlable.setText("")
        self.resultlable.setObjectName("resultlable")
        self.scrollArea_2.setWidget(self.scrollAreaWidgetContents_2)

        self.retranslateUi(video)
        QtCore.QMetaObject.connectSlotsByName(video)

    def retranslateUi(self, video):
        _translate = QtCore.QCoreApplication.translate
        video.setWindowTitle(_translate("video", "Dialog"))
        self.upload.setText(_translate("video", "上传"))
        self.back.setText(_translate("video", "返回"))

