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
        mask = cv.bilateralFilter(mask, 0, 50, 10)  # 对掩膜进行双边模糊，降噪
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


def bi_shift(path):  # 高斯双边滤波与均值迁移滤波
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


def backprograme(roi_p, img_p):  # 直方图反向投影，可以用来提取某一颜色的对象或区域
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


def match(path_1, path_2):  # 模板匹配
    temple = cv.imread(path_1)
    target = cv.imread(path_2)
    tl, tw = temple.shape[: 2]
    res = cv.matchTemplate(target, temple, cv.TM_CCOEFF_NORMED)
    minval, maxval, minloc, maxloc = cv.minMaxLoc(res)
    # lu: 左上角点，rd：右下角点
    lu = maxloc  # 如果为 cv.TM_SQDIFF 系列的方法，左上角点应为minloc，其他的方法为maxloc
    rd = [lu[0] + tw, lu[1] + tl]
    cv.rectangle(target, lu, rd, [0, 0, 255], 2)  # 最后一个参数2为笔画粗细
    cv.imshow("match", target)


def threshold(path):  # 利用阈值二值化
    img = cv.imread(path)
    gary = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 下面是二值化，中间的参数130为自设阈值，加上”|“和之后的为自动设定阈值，ret为阈值，bingary为二值化后的图像
    ret, bingary = cv.threshold(gary, 130, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    gary = cv.bilateralFilter(gary, 0, 20, 10)  # 双边滤波后局部二值化效果更好
    adapt = cv.adaptiveThreshold(gary, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)  # 局部二值化
    print(ret)
    '''    # 调节窗口大小，必须先声明窗口，再设置大小，再显示窗口
    cv.namedWindow("res", 0)
    cv.namedWindow("adaptres", 0)
    cv.resizeWindow("res", 347, 462)
    cv.resizeWindow("adaptres", 347, 462)
    '''
    cv.imshow("res", bingary)
    cv.imshow("adaptres", adapt)


def pyrdown(path):         # 高斯金字塔，向下采样，pyrdown函数返回值类型为numpy.ndarray
    img = cv.imread(path)
    img = cv.resize(img, (512, 512))
    level = 3
    tem = img.copy()
    pyrdown_img = []
    for i in range(level):
        res = cv.pyrDown(tem)
        pyrdown_img.append(res)
        tem = res.copy()
        cv.imshow(str(i), res)
    return pyrdown_img


def laplace(path):      # 由高斯金字塔计算生成拉普拉斯金字塔，lpls为生成的拉普拉斯金字塔，图像尺寸要求2^n，长和宽一致
    img = cv.imread(path)
    img = cv.resize(img, (512, 512))
    pyrdown_img = pyrdown(path)
    level = len(pyrdown_img)
    for i in range(level-1, -1, -1):
        if i-1 < 0:
            expand = cv.pyrUp(pyrdown_img[i], dstsize=img.shape[:2])       # dstsize,是向上采样后生成的图像大小，这里设置成和要做相减运算的图像一样大
            lpls = cv.subtract(img, expand)
        else:
            expand = cv.pyrUp(pyrdown_img[i], dstsize=pyrdown_img[i-1].shape[:2])
            lpls = cv.subtract(pyrdown_img[i-1], expand)
        cv.imshow("lpls"+str(3-i), lpls)


def image_edge(path):   # 利用图像梯度提取图像边缘，sobel与拉普拉斯算子
    img = cv.imread(path)
    # sobel算子，sobel不理想可以用scharr
    grad_x = cv.Sobel(img, cv.CV_32F, 1, 0)
    grad_y = cv.Sobel(img, cv.CV_32F, 0, 1)
    gradx = cv.convertScaleAbs(grad_x)      # 这个函数是通过线性变换，把数据转化成8位【uint8】，符合像素的0-255取值范围
    grady = cv.convertScaleAbs(grad_y)
    grad = cv.addWeighted(gradx, 0.5, grady, 0.5, 0)
    # 下边为拉普拉斯
    res = cv.Laplacian(img, cv.CV_32F)
    rest = cv.convertScaleAbs(res)
    # 下边为自己定义算子，用拉普拉斯为例
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])       # 此处定义的卷积核实际为4邻域拉普拉斯算子
    res1 = cv.filter2D(img, cv.CV_32F, kernel)      # 前边对图像卷积运算时，第二个参数为-1
    rest1 = cv.convertScaleAbs(res1)
    # 效果显示
    cv.imshow("grad", grad)
    cv.imshow("laplace", rest)
    cv.imshow("define", rest1)


def canny_edge(path):       # canny边缘提取
    img = cv.imread(path)
    blur = cv.GaussianBlur(img, [5, 5], 0)          # 提前模糊有必要，可以很好的提高Canny提取边缘的效果
    gary = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)     # 可以不用转化成灰度图，转与不转提取出的边缘区别不大
    dst = cv.Canny(gary, 50, 150)
    cv.imshow("canny", dst)


def line_detect(path):      # 霍夫直线检测
    img = cv.imread(path)
    source = cv.Canny(img, 40, 85)       # 直线检测前必须先提取边缘
    lines = cv.HoughLinesP(source, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(img, [x1, y1], [x2, y2], [0, 0, 255], 2)
    cv.imshow("line", img)
    cv.imshow("source", source)


def circle_detect(path):        # 霍夫圆检测，对噪声非常敏感，需要进行均值迁移滤波，调节参数达到最好效果
    img = cv.imread(path)
    dst = cv.pyrMeanShiftFiltering(img, 10, 100)
    dst = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(dst, cv.HOUGH_GRADIENT, 1, 20, None, 50, 30)
    # print(type(circles))      # 调试需要，看参数类型
    # print(circles)
    for i in circles[0]:
        cv.circle(img, [int(i[0]), int(i[1])], int(i[2]), [0, 0, 255], 2)        # 圆心半径都必须是整数emmmm
        cv.circle(img, [int(i[0]), int(i[1])], 2, [255, 0, 0], 2)
    cv.imshow("canny", dst)
    cv.imshow("img", img)


def counters_find(path):     # 轮廓发现
    img = cv.imread(path)
    # blur = cv.GaussianBlur(img, (3, 3), 0)
    gary = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gary = cv.bilateralFilter(gary, 0, 20, 10)
    # ret, bingard = cv.threshold(gary, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)        # 二值化函数
    binary = cv.adaptiveThreshold(gary, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 25, 10)
    # 上面都是进行调试得到一个比较好的二值化图形，方便进行轮廓发现，不同的方法适用于不同的图像
    counters, heriachy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, counters, -1, (0, 0, 255), 2)
    ''' 下边是用循环显示，一般用上边的方法把counteridx赋值-1即可输出所有轮廓
    for i, counter in enumerate(counters):
        cv.drawContours(img, counters, i, (0, 0, 255), 2)
    '''
    cv.imshow("bingard", binary)
    cv.imshow("counter", img)


def counter_oprate(path):       # 对轮廓进行操作
    img = cv.imread(path)
    img = cv.resize(img, (int(img.shape[0] / 2), int(img.shape[1] / 2)))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    counters, heriachy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for i, counter in enumerate(counters):
        x, y, w, h = cv.boundingRect(counter)       # 外接矩形
        mm = cv.moments(counter)
        cx = int(mm['m10'] / mm['m00'])
        cy = int(mm['m01'] / mm['m00'])
        cv.circle(img, (cx, cy), 3, (0, 0, 255), -1)    # 重心
        cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # cv.drawContours(img, counters, -1, (0, 0, 255), 2)
    cv.imshow("img", img)


def erode_dilate(path):     # 腐蚀和膨胀
    img = cv.imread(path)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    '''   这是对灰度图像、二值图像进行膨胀和腐蚀
    gary = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gary, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    erode_img = cv.erode(binary, kernel)
    dilate_img = cv.dilate(binary, kernel)
    '''
    # 下边是直接对彩色图像进行膨胀和腐蚀
    erode_img = cv.erode(img, kernel)
    dilate_img = cv.dilate(img, kernel)
    cv.imshow("erode", erode_img)
    cv.imshow("diliate", dilate_img)


def open_close(path):       # 图像开闭操作，开操作是先腐蚀，再膨胀，可以去除噪声，提取特定方向的线条等；闭操作是先膨胀再腐蚀，填充闭合区域
    img = cv.imread(path)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    gary = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gary, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    open = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    close = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
    cv.imshow("open", open)
    cv.imshow("close", close)


def watershed(path):        # 分水岭算法!!!!!前边进行形态学操作时应该根据图像二值化图的实际情况进行，目的是为了确定前景和背景
    img = cv.imread(path)
    # 二值化找到所有物体
    blur = cv.pyrMeanShiftFiltering(img, 10, 100)
    gary = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gary, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # 进行形态学操作，先去除图像中的白噪声，用开操作，再去除硬币内部的黑色区域，应该用闭操作
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
    close = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=2)
    dilate = cv.morphologyEx(close, cv.MORPH_DILATE, kernel)    # 膨胀操作，把边界向外扩展到背景中，则剩下的区域一定是背景
    # 距离变换,因为硬币相连，使用开闭、腐蚀膨胀不能进一步分开，故采用距离变换
    dist = cv.distanceTransform(close, cv.DIST_L2, 3)
    dist_output = cv.normalize(dist, 0, 1, cv.NORM_MINMAX)  # 只是归一化显示函数效果，变化到0，1之间更明显
    # 距离变换结束，进行了进一步的分割，但仍相连，下面再用二值化函数，以距离变换结果最大值的0.6为阈值进行二值化，使硬币彻底分开
    ret, surface = cv.threshold(dist, dist.max()*0.6, 255, cv.THRESH_BINARY)    # 这一步得到的surface肯定是硬币，作为前景
    fg = np.uint8(surface)      # 确保值在0到255
    # 确定未知区域：膨胀后的减去确定是硬币的fg
    unknown = cv.subtract(dilate, fg)
    ret, marks = cv.connectedComponents(fg)
    marks += 1      # 用上面函数得到的marks,会把前景图fg中背景标记为0，确定的硬币区域从1开始编号，分水岭算法中0是未知区域，marks整体加一确保没有0
    # 上边的marks里确定是硬币区域是大于1的编号，剩下作为背景的区域为1，这部分区域包括真的背景和未确定部分，未确定的区域即为上面的unknown
    marks[unknown == 255] = 0       # 这一步把背景标记中的未知区域(unknown)的标记改为0，就使确定的背景全标记为1，未知为0，确定前景为大于1的编号
    # 在分水岭算法中，未知区域标记为0，背景为1，前景为大于1的编号
    # ——————上面的这些步骤就生成了分水岭算法所需的mark，下面进行分水岭算法
    # save(marks, 1)
    marks = cv.watershed(img, marks)        # 分水岭算法中边界被标记为-1
    # 下面这部分简单的处理了算法会把图像边界也标记为前景边界的问题，人为改成1，更严谨的做法是进行判断，把图像边界上标为-1的值改为1
    marks[0] = np.ones(marks[0].shape)
    marks[len(marks)-1] = np.ones(marks[len(marks)-1].shape)
    for i, mark in enumerate(marks):
        mark[0] = 1
        mark[len(mark)-1] = 1
    # ————————————————————————————————
    img[marks == -1] = [0, 0, 255]
    # save(marks, 2)
    cv.imshow("1", img)


def save(lis, i):       # 输出分水岭中的markers用
    p = "D:/Study/pyimagehandle/4/" + str(i) + ".txt"
    with open(p, 'w') as f:
        for i, s in enumerate(lis):
            for j in range(len(s)):
                f.write(str(s[j]))
            f.write('\n')


p1 = "D:/Study/pyimagehandle/4/2.jpg"
p2 = "D:/Study/pyimagehandle/1/4.jpg"
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
# match(p2, p1)
# threshold(p1)
# pyrdown(p1)
# laplace(p1)
# image_edge(p1)
# canny_edge(p1)
# line_detect(p1)
# circle_detect(p1)
# counters_find(p1)
# counter_oprate(p1)
# erode_dilate(p1)
# open_close(p1)
watershed(p1)
cv.waitKey(0)
