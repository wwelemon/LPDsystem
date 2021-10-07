import hashlib
import sys
from ui.App import *
from ui.register import *
from ui.test2 import *
from PyQt5.QtWidgets import QApplication,QMainWindow,QDialog,QMessageBox
import sqlite3


def hash(src):
    """
    哈希md5加密方法
    :param src: 字符串str
    :return:
    必须加上@staticmethod
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
    # self.conn = sqlite3.connect("user.db")  # 使用其他数据库的话此处和import模块需要修改
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
                  child1.show()
                  self.close()
              else:
                  QMessageBox.about(self, "错误", "密码不正确")

  def click(self):
      register.show()
      self.close()


class registerWindow(QDialog):
  def __init__(self):
    QDialog.__init__(self)
    self.register = Ui_Dialog()
    self.register.setupUi(self)
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


class childWindow1(QDialog):
  def __init__(self):
    QDialog.__init__(self)
    self.child=Ui_Dialog1()
    self.child.setupUi(self)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    conn = sqlite3.connect("user.db")
    c_sqlite = conn.cursor()

    window = parentWindow()
    register = registerWindow()
    child1 = childWindow1()

    window.show()
    sys.exit(app.exec_())
