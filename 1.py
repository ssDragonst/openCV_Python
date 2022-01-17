import cv2 as cv
import numpy as np  ##


def video():    #视频读取
    capture = cv.VideoCapture(0)
    while True:
        re, frame = capture.read()
        frame = cv.flip(frame, 1)
        cv.imshow("video", frame)
        c = cv.waitKey(1)  # 刷新频率，ms
        if c == 27:  # esc键码
            break


def get_image(path):
    image = cv.imread(path)
    cv.namedWindow("1", cv.WINDOW_AUTOSIZE)
    cv.imshow("1", image)
    cv.waitKey(0)


def color_space(path):
    img = cv.imread(path)
    cv.imshow("began", img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)


def sth_extract():  # 颜色提取
    capture = cv.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        frame = cv.flip(frame, 1)
        if not ret:
            break
        cv.imshow("video", frame)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_hsv = np.array([100, 43, 46])
        upper_hsv = np.array([124, 255, 255])
        mask = cv.inRange(hsv, lower_hsv, upper_hsv)
        dst = cv.bitwise_and(frame, frame, mask=mask)
        cv.imshow("mask", mask)
        cv.imshow("track", dst)
        c = cv.waitKey(1)
        if c > 0:
            break


def contrast_bright(path, c, b):  # 对比度和亮度调节
    image = cv.imread(path)
    l, w, ch = image.shape
    blank = np.zeros([l, w, ch], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1 - c, b)
    cv.imshow("ok", dst)
    cv.imshow("before", image)
    cv.waitKey(0)


def floodfill():  # 泛洪填充
    image = np.zeros([400, 400, 3], np.uint8)
    image[100:300, 100:300] = 255
    mask = np.ones([402, 402, 1], np.uint8)
    mask[100:301, 100:301] = 0
    cv.floodFill(image, mask, (200, 200), (0, 0, 255), cv.FLOODFILL_MASK_ONLY)
    cv.imshow("floodfill", image)
    cv.waitKey(0)


def blur_demo(path):  # 图像模糊
    image = cv.imread(path)
    dst = cv.blur(image, (5, 5))
    cv.imshow("blur", dst)
    cv.imshow("image", image)
    cv.waitKey(0)


def selfdef_blur(path):  # 自定义卷积模板，kernel为卷积模板
    image = cv.imread(path)
    #    kernel = np.ones([5,5],np.float32)/25      #模糊卷积模板
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)  # 锐化卷积模板
    dst = cv.filter2D(image, -1, kernel)
    cv.imshow("image", image)
    cv.imshow("bulr", dst)
    cv.waitKey(0)


p = "D:/Study/pyimagehandle/1/1.jpg"
# sth_extract()
# cv.waitKey(0)
# contrast_bright(p, 1, 50)
# floodfill()
# blur_demo(p)
selfdef_blur(p)
