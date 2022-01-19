import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def video():  # 视频读取
    capture = cv.VideoCapture(0)
    while True:
        re, frame = capture.read()
        frame = cv.flip(frame, 1)
        cv.imshow("video", frame)
        c = cv.waitKey(1)  # 刷新频率，ms
        if c == 27:  # esc键码
            break


def imgcreat():
    image = np.zeros([400, 400, 3], np.uint8)
    image[:, :, 0] = np.ones([400, 400]) * 255
    #    image=np.ones([400,400,3])*255
    cv.imshow("2", image)


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
        mask = cv.bilateralFilter(mask, 0, 50, 10)      # 对掩膜进行双边模糊，降噪
        # 卷积改善效果
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, [7, 7])
        mask = cv.filter2D(mask, -1, kernel)
        #
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


def judge(c):
    if c < 0:
        return 0
    elif c > 255:
        return 255
    else:
        return c


def noise(path):  # 添加噪声，对图像的高斯模糊调用 cv.GaussianBlur
    image = cv.imread(path)
    l, w, c = image.shape
    for i in range(l):
        for j in range(w):
            s = np.random.normal(0, 20, 3)
            b = image[i, j, 0]
            g = image[i, j, 1]
            r = image[i, j, 2]
            image[i, j, 0] = judge(b + s[0])
            image[i, j, 1] = judge(g + s[1])
            image[i, j, 2] = judge(r + s[2])
    cv.imshow("noise", image)
    cv.imshow("image", cv.imread(path))
    return image
    # cv.waitKey(0)


def bi_shift(path):  # 高斯双边模糊与均值迁移模糊
    image = cv.imread(path)
    dst1 = cv.bilateralFilter(image, 0, 20, 10)
    dst2 = cv.pyrMeanShiftFiltering(image, 10, 50)
    cv.imshow("bi", dst1)
    cv.imshow("img", image)
    cv.imshow("shift", dst2)
    cv.waitKey(0)


def image_hist(path):  # 图像直方图
    image = cv.imread(path)
    color = ('b', 'g', 'r')
    for i, cl in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=cl)
        plt.xlim([0, 256])
    cv.imshow("img", image)
    plt.show()


def create_rgb_hist(image):  # 彩色直方图
    h, w, ch = image.shape
    rgbhist = np.zeros([16 * 16 * 16, 1], np.float32)
    bsize = 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int(b / bsize) * 16 * 16 + np.int(g / bsize) * 16 + np.int(
                r / bsize)  # 下降成16个bin，由256*256*256降维4096
            #    print('index:', index)
            rgbhist[np.int(index), 0] += 1
    return rgbhist


def hist_compare(p_1, p_2):  # 直方图比较
    img1 = cv.imread(p_1)
    img2 = cv.imread(p_2)
    hist1 = cv.calcHist([img1.ravel()], [0], None, [256], [0, 256])
    hist2 = cv.calcHist([img2.ravel()], [0], None, [256], [0, 256])
    #    hist1 = create_rgb_hist(img1)
    #    hist2 = create_rgb_hist(img2)
    m1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    m2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    print("巴氏距离： %s  相关系数： %s" % (m1, m2))


def backprograme(roi_p, img_p):  # 直方图反向投影
    roi = cv.imread(roi_p)
    img = cv.imread(img_p)
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv_roi_hist = cv.calcHist([hsv_roi], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv.normalize(hsv_roi_hist, hsv_roi_hist, 0, 255, cv.NORM_MINMAX)  # 直方图归一化，把直方图的值线性调整到指定范围
    dst = cv.calcBackProject([hsv_img], [0, 1], hsv_roi_hist, [0, 180, 0, 256], 1)
    # 下面的卷积把分散的点连到一起
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, [5, 5])  # 生成卷积核
    dst = cv.filter2D(dst, -1, kernel)
    # 把上面处理后的二值图像作为掩膜，进行按位操作
    dst1 = cv.bitwise_and(img, img, mask=dst)
    cv.imshow("1", dst1)


def match(path_1, path_2):      # 模板匹配
    temple = cv.imread(path_1)
    target = cv.imread(path_2)
    tl, tw = temple.shape[: 2]
    res = cv.matchTemplate(target, temple, cv.TM_CCOEFF_NORMED)
    minval, maxval, minloc, maxloc = cv.minMaxLoc(res)
    # lu: 左上角点，rd：右下角点
    lu = maxloc     # 如果为 cv.TM_SQDIFF 系列的方法，左上角点应为minloc，其他的方法为maxloc
    rd = [lu[0]+tw, lu[1]+tl]
    cv.rectangle(target, lu, rd, [0, 0, 255], 2)        # 最后一个参数2为笔画粗细
    cv.imshow("match", target)


p1 = "D:/Study/pyimagehandle/3/1.png"
p2 = "D:/Study/pyimagehandle/3/2.png"
# sth_extract()
# cv.waitKey(0)
# contrast_bright(p, 1, 50)
# floodfill()
# blur_demo(p)
# selfdef_blur(p)
# bi_shift(p1)
# image_hist(p1)
# hist_compare(p1, p2)
# backprograme(p2, p1)
match(p2, p1)
cv.waitKey(0)
