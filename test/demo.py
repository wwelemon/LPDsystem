from login_ui import LoginUi
from ui.test2 import *
from login_ui import *
import sys
import sqlite3
import hashlib
import time
from PyQt5.QtWidgets import QMainWindow,QDialog,QMessageBox

class childWindow(QDialog):
  def __init__(self):
    QDialog.__init__(self)
    self.child=Ui_Dialog1()
    self.child.setupUi(self)

class parentWindow(QMainWindow):
  def __init__(self):
    QMainWindow.__init__(self)
    self.main_ui = Ui_MainWindow()
    self.main_ui.setupUi(self)
    self.main_ui.registerbutton.clicked.connect(lambda: self.back())

  def back(self):
      ui.show()
      self.close()

# 继承界面
class LoginLogic(LoginUi):
    def __init__(self):
        super(LoginLogic, self).__init__()
        self.conn = sqlite3.connect("user.db")  # 使用其他数据库的话此处和import模块需要修改
        # 此处改变密码输入框lineEdit_password的属性，使其不现实密码
        self.lineEdit_password.setEchoMode(QtWidgets.QLineEdit.Password)
        # qt的信号槽机制，连接按钮的点击事件和相应的方法
        self.pussButton_signin.clicked.connect(lambda: self.sign_in())
        self.pussButton_signup.clicked.connect(lambda: self.sign_up())


    @staticmethod
    def hash(src):
        """
        哈希md5加密方法
        :param src: 字符串str
        :return:
        """
        src = (src + "请使用私钥加密").encode("utf-8")
        print(src)
        m = hashlib.md5()
        m.update(src)
        return m.hexdigest()

    def sign_in(self):
        """
        登陆方法
        :return:
        """
        c_sqlite = self.conn.cursor()
        user_name = self.lineEdit_user.text()
        user_password = self.lineEdit_password.text()

        if user_name == "" or user_password == "":
            QMessageBox.about(self, "提示", "请输入用户名和密码")
        else:
            c_sqlite.execute("""SELECT password FROM user WHERE name = ?""", (user_name,))
            password = c_sqlite.fetchall()
            if not password:
                QMessageBox.about(self, "提示", "此用户未注册")
            else:
                print('self.hash(user_password),password[0][0]',self.hash(user_password),password[0][0])
                if self.hash(user_password) == password[0][0]:
                    QMessageBox.about(self, "提示", "登陆成功")
                    time.sleep(1)
                    # self.open_main_window()
                    parentWindow.show()
                    self.close()
                else:
                    QMessageBox.about(self,"错误","密码不正确")

    def sign_up(self):
        """
        注册方法
        :return:
        """
        c_sqlite = self.conn.cursor()
        user_name = self.lineEdit_user.text()
        user_password = self.lineEdit_password.text()
        if user_name == "" or user_password == "":
            QMessageBox.about(self, "提示", "请输入用户名和密码")
        else:
            user_password = self.hash(user_password)
            c_sqlite.execute("""SELECT password FROM user WHERE name = ?""", (user_name,))
            if not c_sqlite.fetchall():
                c_sqlite.execute("""INSERT INTO user VALUES (NULL ,?,?)""", (user_name, user_password))
                self.conn.commit()
                QMessageBox.about(self, "提示", "注册成功")
            else:
                QMessageBox.about(self, "提示", "用户名重复")

    def open_main_window(self):
        """
        此处添加打开另一个窗口的程序
        :return:
        """
        child.show()
        # self.open()
        print("打开另一个窗口")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = LoginLogic()
    child = childWindow()
    parentWindow = parentWindow()
    # ui.pussButton_signin.clicked.connect(lambda: ui.sign_in())
    # ui.pussButton_signup.clicked.connect(lambda: ui.sign_up())
    ui.show()

    sys.exit(app.exec_())
