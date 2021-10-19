# -*- codeing = utf-8 -*-
# @Time:2021/3/27  10:29 基于rgb颜色,无深度学习
# @Author:王鹏海
# @File:detect.py
# @Software:PyCharm
import time
import numpy as np
import cv2
import os
from numpy.linalg import norm

import SVM_Train

Blue = 138
Green = 63
Red = 23
THRESHOLD = 50
ANGLE = -45
MIN_AREA = 1500
MAX_AREA = 5000
LICENSE_WIDTH = 440
LICENSE_HIGH = 140
resize_h = 720


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)
        # 不能保证包括所有省份

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 调用svm
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()



def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


def predict_svm1(img,plate_svm):

    digit_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    digit_img = cv2.resize(digit_img, (136, 36))

    chars_train = preprocess_hog([digit_img])

    resp = plate_svm.predict(chars_train)

    return resp[0]


# 绘图展示(后期放入工具类中)
def cv_show(name, img):
    cv2.namedWindow(name, 0)
    # print(img.shape)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_files(path, all_files):
    # 首先遍历当前目录所有文件及文件夹
    file_list = os.listdir(path)
    # 准备循环判断每个元素是否是文件夹还是文件，是文件的话，把名称传入list，是文件夹的话，递归
    for file in file_list:
        # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
        cur_path = os.path.join(path, file)
        # 判断是否是文件夹
        if os.path.isdir(cur_path):
            show_files(cur_path, all_files)
        else:
            if cur_path.find(".jpg") > 0:
                all_files.append(cur_path)
    return all_files


def computeSafeRegion(shape, bounding_rect):
    top = bounding_rect[1]  # y
    bottom = bounding_rect[1] + bounding_rect[3]  # y +  h
    left = bounding_rect[0]  # x
    right = bounding_rect[0] + bounding_rect[2]  # x +  w

    min_top = 0
    max_bottom = shape[0]
    min_left = 0
    max_right = shape[1]

    # print "computeSateRegion input shape",shape
    if top < min_top:
        top = min_top
        # print "tap top 0"
    if left < min_left:
        left = min_left
        # print "tap left 0"

    if bottom > max_bottom:
        bottom = max_bottom
        # print "tap max_bottom max"
    if right > max_right:
        right = max_right

    return [left, top, right - left, bottom - top]


def cropped_from_image(image, rect):
    x, y, w, h = computeSafeRegion(image.shape, rect)
    return image[y:y + h, x:x + w]


def verify_scale(rotate_rect):
    error = 0.4
    aspect = 3  # 4.7272
    min_area = 500
    max_area = 150 * (150 * aspect)
    min_aspect = aspect * (1 - error)
    max_aspect = aspect * (1 + error)
    theta = 30

    # 宽或高为0，不满足矩形直接返回False
    if rotate_rect[1][0] == 0 or rotate_rect[1][1] == 0:
        return False

    r = rotate_rect[1][0] / rotate_rect[1][1]
    r = max(r, 1 / r)
    # print('长宽比：',r)
    area = rotate_rect[1][0] * rotate_rect[1][1]
    if area > min_area and area < max_area and r > min_aspect and r < max_aspect:
        # 矩形的倾斜角度在不超过theta
        if ((rotate_rect[1][0] < rotate_rect[1][1] and rotate_rect[2] >= -90 and rotate_rect[2] < -(90 - theta)) or
                (rotate_rect[1][1] < rotate_rect[1][0] and rotate_rect[2] > -theta and rotate_rect[2] <= 0)):
            return True
    return False


# 简单移动平均法
def ma(list2):
    n = len(list2)
    sum = 0
    for i in list2:
        sum += i
    result = sum / n
    return result


def answer1(list1, n):
    # 简单移动平均法
    listMA = []  # 简单移动平均值的列表
    for i in range(n - 1, len(list1)):
        # print(i)
        list2 = (list1[i - (n - 1):i + 1])
        listMA.append(ma(list2))
    # print("简单移动平均值的列表：{}".format(listMA))
    # # 最后的移动平均值可做为下一个数的预测
    # x = listMA[-1]
    # print("下一个数的预测:{}".format(x))
    # # 画图
    # plt.scatter(list(range(len(listMA))), listMA)
    #
    # plt.show()
    return listMA


def seekopp(list_2, i):
    for j in range(i - 1, -1, -1):
        if list_2[j] == 0:
            continue
        elif list_2[j] > 0:
            return 1
        elif list_2[j] < 0:
            return 0


def local_maximum(list_1):
    a = len(list_1)
    if a == 0:
        return 'error'
    if a == 1:
        return list_1
    if a == 2:
        if list_1[0] > list_1[1]:
            return list_1[0]
        elif list_1[0] < list_1[1]:
            return list_1[1]
        else:
            return list_1
    if a > 2:
        list_2 = []
        index_1 = []
        for i in range(0, a - 1):
            list_2.append(list_1[i + 1] - list_1[i])
        b = len(list_2)
        if list_2[0] < 0:
            index_1.append(0)
        for i in range(0, b - 1):
            if list_2[i + 1] < 0:
                if list_2[i] > 0:
                    index_1.append(i + 1)
                elif list_2[i] == 0:
                    if seekopp(list_2, i):
                        index_1.append(i + 1)
                else:
                    continue
            else:
                continue
        list_3 = []
        for i in index_1:
            list_3.append(list_1[i])
        return list_3


def local_minimum(list_1):
    a = len(list_1)
    if a > 2:
        list_2 = []
        index_1 = []
        for i in range(0, a - 1):
            list_2.append(list_1[i + 1] - list_1[i])
        b = len(list_2)
        if list_2[0] > 0:
            index_1.append(0)
        for i in range(0, b - 1):
            if list_2[i + 1] > 0:
                if list_2[i] < 0:
                    index_1.append(i + 1)
                elif list_2[i] == 0:
                    if seekopp(list_2, i):
                        index_1.append(i + 1)
                else:
                    continue
            else:
                continue
        list_3 = []
        for i in index_1:
            list_3.append(list_1[i])
        return list_3


def max_min(local_maximums, local_minimums, l):
    list_max_min = []

    for i in range(0, len(local_maximums), 1):
        if len(local_maximums) > len(local_minimums):
            if i < len(local_minimums):
                k = local_maximums[i] - local_minimums[i]
                if k > l:
                    list_max_min.append(k)
                k1 = local_maximums[i + 1] - local_minimums[i]
                if k1 > l:
                    list_max_min.append(k1)
        if len(local_maximums) < len(local_minimums):
            if i < len(local_maximums):
                k = local_maximums[i] - local_minimums[i]
                if k > l:
                    list_max_min.append(k)
                k1 = local_maximums[i] - local_minimums[i + 1]
                if k1 > l:
                    list_max_min.append(k1)
        if len(local_maximums) == len(local_minimums):
            if i == 0:
                k = local_maximums[i] - local_minimums[i]
                if k > l:
                    list_max_min.append(k)
            else:
                k = local_maximums[i] - local_minimums[i - 1]
                if k > l:
                    list_max_min.append(k)
                k1 = local_maximums[i] - local_minimums[i]
                if k1 > l:
                    list_max_min.append(k1)
    return list_max_min


def color_filter(cropped, size, channel, fall, nums):
    # RGB过滤
    img_rgb = cv2.cvtColor(cropped, cv2.COLOR_HSV2RGB)

    height = int(img_rgb.shape[0] / 2)
    B_COLOR = []
    # H=[]
    for i in range(0, cropped.shape[1], 1):
        B_COLOR.append(img_rgb[height, i, channel])

    listMA = answer1(B_COLOR, size)  # 简单移动平均法
    local_maximums = local_maximum(listMA)
    local_minimums = local_minimum(listMA)
    # print('H',H)
    list_max_min = max_min(local_maximums, local_minimums, fall)
    # print('list_max_min', len(list_max_min), list_max_min)
    # print('----------------------------------')
    if len(list_max_min) < nums:
        return False
    else:
        return True

def pre_process(img):

    img_aussian = cv2.GaussianBlur(img, (5, 5), 1)

    img_median = cv2.medianBlur(img_aussian, 3)

    gray_img = cv2.cvtColor(img_median, cv2.COLOR_BGR2GRAY)

    sobel_img = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0, ksize=3)
    sobel_img = cv2.convertScaleAbs(sobel_img)

    hsv_img = cv2.cvtColor(img_median, cv2.COLOR_BGR2HSV)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]
    # 黄色色调区间[26，34],蓝色色调区间:[100,124]
    blue_img = (((h > 100) & (h < 124))) & ((s > 100) & (s < 255)) & ((v > 50) & (v < 255))
    blue_img = blue_img.astype('float32')

    mix_img = np.multiply(sobel_img, blue_img)
    # cv_show('mix_img', mix_img)
    mix_img = mix_img.astype(np.uint8)

    ret, binary_img = cv2.threshold(mix_img, 1, 255, cv2.THRESH_BINARY)
    # cv_show('binary_img', binary_img)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
    close_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    # cv_show('close_img', close_img)

    image, contours, hierarchy = cv2.findContours(close_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def points(rect):
    car_contours = []
    car_contours.append(rect)  # rect是minAreaRect的返回值，根据minAreaRect的返回值计算矩形的四个点
    box = cv2.boxPoints(rect)  # box里面放的是最小矩形的四个顶点坐标
    box = np.int0(box)  # 取整
    left_point_x = np.min(box[:, 0])
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])
    left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
    right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
    top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
    bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
    return left_point_x, top_point_y, right_point_x, left_point_x, bottom_point_y, top_point_y


def detect(img):
    global lx, ly, ry, rx, flg
    contours = pre_process(img)

    if len(contours) < 0:
        return img
    else:
        cropped_images = []

        for cnt in contours:
            # 框选 生成最小外接矩形 返回值（中心(x,y), (宽,高), 旋转角度）rect[0]：矩形中心点坐标；rect[1]：矩形的高和宽；rect[2]：矩形的旋转角度
            rect = cv2.minAreaRect(cnt)
            if verify_scale(rect):
                left_point_x, top_point_y, right_point_x, left_point_x, bottom_point_y, top_point_y = points(rect)
                cropped = cropped_from_image(img, (
                    int(left_point_x), int(top_point_y), abs(right_point_x - left_point_x+10),
                    abs(bottom_point_y - top_point_y+10)))

                # RGB过滤
                color_filter2 = color_filter(cropped, 15, 0, 4, 10)
                if color_filter2:
                    # cv_show('cropped', cropped)
                    cropped_images.append(cropped)
                    cv2.rectangle(img, (int(left_point_x), int(top_point_y)),
                                  (int(right_point_x), int(bottom_point_y)), (0,0, 255), 3)

    return cropped_images,img,cropped


def identification(card_imgs):
    # 识别车牌中的字符
    model = SVM(C=1, gamma=0.5)
    if os.path.exists("svm.dat"):
        model.load("svm.dat")
    else:
        raise FileNotFoundError('svm.dat')
    modelchinese = SVM(C=1, gamma=0.5)
    if os.path.exists("svmchinese.dat"):
        modelchinese.load("svmchinese.dat")
    else:
        raise FileNotFoundError('svmchinese.dat')
    predict_result = []
    SZ = 20

    for i in range(0,len(card_imgs)):
        card_img = card_imgs[i]
        # old_img = card_img
        # 做一次锐化处理
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化
        card_img = cv2.filter2D(card_img, -1, kernel=kernel)
        # cv2.imshow("custom_blur", card_img)

        # RGB转GARY
        gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray_img', gray_img)

        # 二值化
        ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 查找水平直方图波峰
        x_histogram = np.sum(gray_img, axis=1)
        # 最小值
        x_min = np.min(x_histogram)
        # 均值
        x_average = np.sum(x_histogram) / x_histogram.shape[0]
        x_threshold = (x_min + x_average) / 2
        wave_peaks = find_waves(x_threshold, x_histogram)
        if len(wave_peaks) == 0:
            continue

        # 认为水平方向，最大的波峰为车牌区域
        wave = max(wave_peaks, key=lambda x: x[1] - x[0])
        gray_img = gray_img[wave[0]:wave[1]]
        #cv2.imshow('gray_img', gray_img)

        # 查找垂直直方图波峰
        row_num, col_num = gray_img.shape[:2]
        # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
        gray_img = gray_img[1:row_num - 1]
        # cv2.imshow('gray_img', gray_img)
        y_histogram = np.sum(gray_img, axis=0)
        y_min = np.min(y_histogram)
        y_average = np.sum(y_histogram) / y_histogram.shape[0]
        y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半

        wave_peaks = find_waves(y_threshold, y_histogram)

        if len(wave_peaks) <= 6:
            #   print(wave_peaks)
            continue

        wave = max(wave_peaks, key=lambda x: x[1] - x[0])
        max_wave_dis = wave[1] - wave[0]
        # 判断是否是左侧车牌边缘
        if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
            wave_peaks.pop(0)

        # 组合分离汉字
        cur_dis = 0
        for i, wave in enumerate(wave_peaks):
            if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                break
            else:
                cur_dis += wave[1] - wave[0]
        if i > 0:
            wave = (wave_peaks[0][0], wave_peaks[i][1])
            wave_peaks = wave_peaks[i + 1:]
            wave_peaks.insert(0, wave)

        # 去除车牌上的分隔点
        point = wave_peaks[2]
        if point[1] - point[0] < max_wave_dis / 3:
            point_img = gray_img[:, point[0]:point[1]]
            if np.mean(point_img) < 255 / 5:
                wave_peaks.pop(2)

        if len(wave_peaks) <= 6:
            # print("peak less 2:", wave_peaks)
            continue
        part_cards = seperate_card(gray_img, wave_peaks)

        # 识别
        for i, part_card in enumerate(part_cards):
            # 可能是固定车牌的铆钉
            if np.mean(part_card) < 255 / 5:
                continue
            w = abs(part_card.shape[1] -SZ) // 2

            # 边缘填充
            part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            # cv2.imshow('part_card', part_card)

            # 图片缩放（缩小）
            part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
            # cv2.imshow('part_card', part_card)
            part_card = SVM_Train.preprocess_hog([part_card])
            if i == 0:  # 识别汉字
                provinces = ["zh_cuan", "川", "zh_e", "鄂", "zh_gan", "赣", "zh_gan1", "甘", "zh_gui", "贵", "zh_gui1", "桂",
                             "zh_hei", "黑",
                             "zh_hu", "沪", "zh_ji", "冀", "zh_jin", "津", "zh_jing", "京", "zh_jl", "吉", "zh_liao", "辽",
                             "zh_lu", "鲁",
                             "zh_meng", "蒙", "zh_min", "闽", "zh_ning", "宁", "zh_qing", "靑", "zh_qiong", "琼", "zh_shan",
                             "陕", "zh_su",
                             "苏", "zh_sx", "晋", "zh_wan", "皖", "zh_xiang", "湘", "zh_xin", "新", "zh_yu", "豫", "zh_yu1",
                             "渝", "zh_yue",
                             "粤", "zh_yun", "云", "zh_zang", "藏", "zh_zhe", "浙"]
                PROVINCE_START = 1000
                resp = modelchinese.predict(part_card)  # 匹配样本
                charactor = provinces[int(resp[0]) - PROVINCE_START]
                # print(charactor)
            else:  # 识别字母
                resp = model.predict(part_card)  # 匹配样本
                charactor = chr(resp[0])
                # print(charactor)
            predict_result.append(charactor)
        break
    return predict_result # 识别到的字符、定位的车牌图像、车牌颜色

# 利用投影法，根据设定的阈值和图片直方图，找出波峰，用于分隔字符
def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks

# 根据找出的波峰，分隔图片，从而得到逐个字符图片
def seperate_card(img, waves):
    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]:wave[1]])
    return part_cards


def cutVideo(videoFileName):
    i = 0
    video = cv2.VideoCapture(videoFileName)  # 读取视频文件
    while video.isOpened():
        ret, frame = video.read()
        i = i + 1
        if not ret:
            print("没有读取到")
            break
        # if i % 20 == 0:
        if i % 1 == 0:
           result,frame,crop = detect(frame)
           cv2.imshow("a",crop)
           key = cv2.waitKey(1) & 0xFF
           if key == ord("q"):
               break
        continue

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    # img_name ="D:/LPDsystem/img/AA662L.jpg"
    # print(type(img_name))
    # img_raw = cv2.imread(img_name)
    # image = detect(img_raw)
    # predict_result = identification(image)
    # print("predict_result", predict_result)

    videoname = "D:/LPDsystem/img/test2.mp4"
    cutVideo(videoname)









