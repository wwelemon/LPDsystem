# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'App.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(795, 597)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setEnabled(True)
        self.frame.setGeometry(QtCore.QRect(100, 0, 601, 531))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(170, 430, 81, 27))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.password = QtWidgets.QLineEdit(self.frame)
        self.password.setGeometry(QtCore.QRect(256, 430, 220, 30))
        self.password.setMinimumSize(QtCore.QSize(217, 0))
        self.password.setObjectName("password")
        self.loginbutton = QtWidgets.QPushButton(self.frame)
        self.loginbutton.setEnabled(True)
        self.loginbutton.setGeometry(QtCore.QRect(380, 478, 93, 28))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(16)
        self.loginbutton.setFont(font)
        self.loginbutton.setObjectName("loginbutton")
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setGeometry(QtCore.QRect(170, 380, 71, 27))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.username = QtWidgets.QLineEdit(self.frame)
        self.username.setGeometry(QtCore.QRect(256, 380, 220, 30))
        self.username.setObjectName("username")
        self.registerbutton = QtWidgets.QPushButton(self.frame)
        self.registerbutton.setGeometry(QtCore.QRect(256, 478, 93, 28))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(16)
        self.registerbutton.setFont(font)
        self.registerbutton.setObjectName("registerbutton")
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(130, 50, 391, 91))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(36)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 795, 26))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)
        self.action23 = QtWidgets.QAction(MainWindow)
        self.action23.setObjectName("action23")
        self.action256 = QtWidgets.QAction(MainWindow)
        self.action256.setObjectName("action256")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "车牌识别系统"))
        self.label_2.setText(_translate("MainWindow", "密   码："))
        self.loginbutton.setText(_translate("MainWindow", "登录"))
        self.label.setText(_translate("MainWindow", "用户名:"))
        self.registerbutton.setText(_translate("MainWindow", "注册"))
        self.label_3.setText(_translate("MainWindow", "智能车牌识别系统"))
        self.action23.setText(_translate("MainWindow", "23"))
        self.action256.setText(_translate("MainWindow", "256"))

