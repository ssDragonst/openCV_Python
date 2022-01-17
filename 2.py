import cv2 as cv
import numpy as np


def imgcreat():
    image = np.zeros([400, 400, 3], np.uint8)
    image[:, :, 0] = np.ones([400, 400]) * 255
    #    image=np.ones([400,400,3])*255
    cv.imshow("2", image)


imgcreat()
cv.waitKey(0)
