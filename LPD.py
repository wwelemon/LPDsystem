import hashlib
import sys,os
from ui.App import *
from ui.register import *
from ui.recognise import *
from ui.image import  *
from ui.video import *
from PyQt5.QtWidgets import QApplication,QMainWindow,QDialog,QMessageBox,QFileDialog
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt,QTimer,pyqtSignal
import sqlite3
import cv2
import numpy as np
from detect5 import detect,identification


def hash(src):
    """
    哈希md5加密方法
    :param src: 字符串str
    :return:
    在类内必须加上@staticmethod
    """
    src = (src + "LPD").encode("utf-8")
    print(src)
    m = hashlib.md5()
    m.update(src)
    return m.hexdigest()

class parentWindow(QMainWindow):
  def __init__(self):
    QMainWindow.__init__(self)
    self.main_ui = Ui_MainWindow()
    self.main_ui.setupUi(self)
    self.setStyleSheet("#MainWindow{border-image:url(img/4.png);}")
    # self.main_ui.registerbutton.setStyleSheet('''QPushButton{border-image:url(img/3.png);}''')
    self.main_ui.registerbutton.setStyleSheet(
        '''QPushButton{background:#F7D674;border-radius:20px;}QPushButton:hover{background:yellow;}''')
    self.main_ui.loginbutton.setStyleSheet(
        '''QPushButton{background:#F7D674;border-radius:20px;}QPushButton:hover{background:yellow;}''')

    # 此处改变密码输入框lineEdit_password的属性，使其不现实密码
    self.main_ui.password.setEchoMode(QtWidgets.QLineEdit.Password)
    # qt的信号槽机制，连接按钮的点击事件和相应的方法
    self.main_ui.loginbutton.clicked.connect(lambda: self.sign_in())
    self.main_ui.registerbutton.clicked.connect(lambda: self.click())

  def sign_in(self):
      """
      登陆方法
      :return:
      """
      user_name = self.main_ui.username.text()
      user_password = self.main_ui.password.text()
      if user_name == "" or user_password == "":
          QMessageBox.about(self, "提示", "请输入用户名和密码")
      else:
          c_sqlite.execute("""SELECT password FROM user WHERE name = ?""", (user_name,))
          password = c_sqlite.fetchall()

          if not password:
              QMessageBox.about(self, "提示", "此用户未注册")
          else:
              if hash(user_password) == password[0][0]:
                  QMessageBox.about(self, "提示", "登陆成功")
                  recognise.show()
                  self.close()
              else:
                  QMessageBox.about(self, "错误", "密码不正确")

  def click(self):
      register.show()
      self.close()


class registerWindow(QDialog):
  def __init__(self):
    QDialog.__init__(self)
    self.register = Ui_registerui()
    self.register.setupUi(self)
    self.setStyleSheet("#registerui{border-image:url(img/4.png);}")
    self.register.backbutton.setStyleSheet(
        '''QPushButton{background:#F7D674;border-radius:20px;}QPushButton:hover{background:yellow;}''')
    self.register.registerbutton.setStyleSheet(
        '''QPushButton{background:#F7D674;border-radius:20px;}QPushButton:hover{background:yellow;}''')

    self.register.password.setEchoMode(QtWidgets.QLineEdit.Password)
    self.register.backbutton.clicked.connect(lambda: self.returnclick())
    self.register.registerbutton.clicked.connect(lambda: self.sign_up())


  def returnclick(self):
      window.show()
      self.close()

  def sign_up(self):
      """
      注册方法
      :return:
      """
      user_name = self.register.username.text()
      user_password = self.register.password.text()
      if user_name == "" or user_password == "":
          QMessageBox.about(self, "提示", "注册失败，请输入用户名和密码")
      else:
          user_password = hash(user_password)
          c_sqlite.execute("""SELECT password FROM user WHERE name = ?""", (user_name,))
          if not c_sqlite.fetchall():
              c_sqlite.execute("""INSERT INTO user VALUES (NULL ,?,?)""", (user_name, user_password))
              self.conn.commit()
              QMessageBox.about(self, "提示", "注册成功")
              window.show()
              self.close()
          else:
              QMessageBox.about(self, "提示", "用户名重复")


class recognise(QDialog):

  def __init__(self,image):
    QDialog.__init__(self)
    self.recognise=Ui_recognise()
    self.recognise.setupUi(self)
    self.setStyleSheet("#recognise{border-image:url(img/5.png);}")
    self.recognise.upload.setStyleSheet(
        '''QPushButton{background:#F7D674;border-radius:20px;}QPushButton:hover{background:yellow;}''')
    self.recognise.back.setStyleSheet(
        '''QPushButton{background:#F7D674;border-radius:20px;}QPushButton:hover{background:yellow;}''')
    self.recognise.upload.clicked.connect(lambda: self.openimage(image))
    self.recognise.back.clicked.connect(lambda: self.returnclick())
    # print("path,filename",path,filename)
    self.recognise.ProjectPath = os.getcwd()
    # self.image = image


  def returnclick(self):

      image.show()
      self.close()


  def openimage(self,image):
      path, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
      filename = path.split('/')[-1]
      print("filename",path)
      self.recognise.img.resize(500,300)
      self.recognise.img.setAlignment(Qt.AlignCenter)
      size = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR).shape
      if size[0] / size[1] > 1.0907:
          w = size[1] * self.recognise.img.height() / size[0]
          h = self.recognise.img.height()
          jpg = QtGui.QPixmap(path).scaled(w, h)
      elif size[0] / size[1] < 1.0907:
          w = self.recognise.img.width()
          h = size[0] * self.recognise.img.width() / size[1]
          jpg = QtGui.QPixmap(path).scaled(w, h)
      else:
          jpg = QtGui.QPixmap(path).scaled(self.recognise.img.width(), self.recognise.img.height())
      self.recognise.img.setPixmap(jpg)
      result,croppedplate = self.vlpr(path)

      image.showimage(path,croppedplate)

      result = ''.join(result)
      print(result)

      if result is not None:
          self.recognise.result.setText(result)

      else:
          self.recognise.result.setText("无法识别")
          QMessageBox.warning(self, "Error", "无法识别此图像！", QMessageBox.Yes)


  def vlpr(self,path):
      # print("1")
      # PR = PlateRecognition()
      # result = PR.VLPR(path)
      img_raw = cv2.imread(path)
      # print("2")
      result,img,croppedplate = detect(img_raw)
      result = identification(result)
      print(result)
      return result,croppedplate



class image(QDialog):

    def __init__(self):
        QDialog.__init__(self)
        self.image = Ui_image()
        self.image.setupUi(self)
        self.image.pushButton.clicked.connect(lambda: self.returnclick())
        self.image.pushButton_2.clicked.connect(lambda: self.videoclick())
    def showimage(self,path,plate):
        size = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR).shape
        if size[0] / size[1] > 1.0907:
            w = size[1] * self.image.origin.height() / size[0]
            h = self.image.origin.height()
            jpg = QtGui.QPixmap(path).scaled(w, h)
        elif size[0] / size[1] < 1.0907:
            w = self.image.origin.width()
            h = size[0] * self.image.origin.width() / size[1]
            jpg = QtGui.QPixmap(path).scaled(w, h)
        else:
            jpg = QtGui.QPixmap(path).scaled(self.image.origin.width(), self.image.origin.height())
        self.image.origin.setPixmap(jpg)


        img = cv2.imread(path)
        img_aussian0 = cv2.GaussianBlur(img, (5, 5), 1)
        img_aussian = cv2.resize(img_aussian0,(200,150),interpolation=cv2.INTER_AREA)
        img_aussian = cv2.cvtColor(img_aussian, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(img_aussian.data, img_aussian.shape[1], img_aussian.shape[0], QtGui.QImage.Format_RGB888)
        self.image.gaussian.setPixmap(QtGui.QPixmap.fromImage(showImage))


        img_median0 = cv2.medianBlur(img_aussian0, 3)
        gray_img0 = cv2.cvtColor(img_median0, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.resize(gray_img0, (200, 150), interpolation=cv2.INTER_AREA)
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
        gray_img = QtGui.QImage(gray_img.data, gray_img.shape[1], gray_img.shape[0], QtGui.QImage.Format_RGB888)
        self.image.gray.setPixmap(QtGui.QPixmap.fromImage(gray_img))


        sobel_img0 = cv2.Sobel(gray_img0, cv2.CV_16S, 1, 0, ksize=3)
        sobel_img = cv2.convertScaleAbs(sobel_img0)
        sobel_img = cv2.resize(sobel_img, (200, 150), interpolation=cv2.INTER_AREA)
        sobel_img = cv2.cvtColor(sobel_img, cv2.COLOR_GRAY2RGB)
        sobel_img = QtGui.QImage(sobel_img.data, sobel_img.shape[1], sobel_img.shape[0], QtGui.QImage.Format_RGB888)
        self.image.sobel.setPixmap(QtGui.QPixmap.fromImage(sobel_img))


        hsv_img = cv2.cvtColor(img_median0, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
        # 黄色色调区间[26，34],蓝色色调区间:[100,124]
        blue_img = (((h > 100) & (h < 124))) & ((s > 100) & (s < 255)) & ((v > 50) & (v < 255))
        blue_img = blue_img.astype('float32')
        mix_img = np.multiply(sobel_img0, blue_img)
        mix_img0 = mix_img.astype(np.uint8)
        mix_img = cv2.resize(mix_img0, (200, 150), interpolation=cv2.INTER_AREA)
        mix_img = cv2.cvtColor(mix_img, cv2.COLOR_GRAY2RGB)
        mix_img = QtGui.QImage(mix_img.data, mix_img.shape[1], mix_img.shape[0], QtGui.QImage.Format_RGB888)
        self.image.blue.setPixmap(QtGui.QPixmap.fromImage(mix_img))


        ret, binary_img = cv2.threshold(mix_img0, 1, 255, cv2.THRESH_BINARY)
        binary_img = cv2.resize(binary_img, (200, 150), interpolation=cv2.INTER_AREA)
        binary_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2RGB)
        binary_img = QtGui.QImage(binary_img.data, binary_img.shape[1], binary_img.shape[0], QtGui.QImage.Format_RGB888)
        self.image.binary.setPixmap(QtGui.QPixmap.fromImage(binary_img))

        print('plate', plate)
        plate = cv2.resize(plate, (200, 60), interpolation=cv2.INTER_AREA)
        plate = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
        plate = QtGui.QImage(plate.data, plate.shape[1], plate.shape[0], QtGui.QImage.Format_RGB888)
        self.image.plate.setPixmap(QtGui.QPixmap.fromImage(plate))

    def returnclick(self):
        recognise.show()
        self.close()

    def videoclick(self):
        video.show()
        self.close()


class video(QDialog):
  def __init__(self):
    QDialog.__init__(self)
    self.video = Ui_video()
    self.video.setupUi(self)
    self.video.upload.clicked.connect(lambda: self.openvideo())
    self.video.back.clicked.connect(lambda: self.returnclick())
    self.video.ProjectPath = os.getcwd()


  def openvideo(self):
      path, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.mp4;;*.avi;;All Files(*)")
      filename = path.split('/')[-1]
      print("filename", path)
      self.video.videoframe.resize(700, 400)
      self.video.videoframe.setAlignment(Qt.AlignCenter)

      cap = cv2.VideoCapture(path)
      while cap.isOpened():
          success, frame = cap.read()
          if not success:
              break
          if success:
              predict_result, frame,croppedplate = detect(frame)
              croppedplate = cv2.resize(croppedplate, (200, 50), interpolation=cv2.INTER_AREA)
              croppedplate = cv2.cvtColor(croppedplate, cv2.COLOR_BGR2RGB)
              croppedplate = QImage(croppedplate.data, croppedplate.shape[1], croppedplate.shape[0], QImage.Format_RGB888)
              self.video.resultlable.setPixmap(QPixmap.fromImage(croppedplate))
              frame = cv2.resize(frame,(700,400),interpolation=cv2.INTER_AREA)
              show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
              showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
              self.video.videoframe.setPixmap(QPixmap.fromImage(showImage))
              # cv2.imshow("Frame", frame)
              cv2.waitKey(1)

      cv2.destroyAllWindows()
      cap.release()

  def returnclick(self):
      image.show()
      self.close()


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    conn = sqlite3.connect("user.db")
    c_sqlite = conn.cursor()

    window = parentWindow()
    register = registerWindow()
    image = image()
    recognise = recognise(image)
    video = video()


    window.show()
    sys.exit(app.exec_())
