import cv2 as cv
import numpy as np


if __name__ == '__main__':
    fname = './pictures/meiyan.jpg'
    k_size_median = 5
    k_size_gauss = 3
    dim = 40
    sigma_s = dim * 2
    sigma_r = dim
    sigma_g = dim/20
    img = cv.imread(fname)
    img_out = cv.medianBlur(img, k_size_median)
    img_out = cv.bilateralFilter(img_out, dim, sigma_r, sigma_s)
    cv.imshow('img', img)
    cv.imshow('img_bi', img_out)
    cv.imwrite('./result/meiyan_bi.jpg', img_out)
    img_sobelX = cv.Sobel(img_out, cv.CV_8U, 1, 0, ksize=3)
    img_sobelX = np.uint8(0.05*img_sobelX)
    cv.imshow('SobelX', img_sobelX)
    img_sobelY = cv.Sobel(img_out, cv.CV_8U, 0, 1, ksize=3)
    img_sobelY = np.uint8(0.05*img_sobelY)
    img_out = cv.add(img_out, img_sobelX)
    img_out = cv.add(img_out, img_sobelY)
    cv.imshow('img_out', img_out)
    cv.imwrite('./result/meiyanResult.jpg', img_out)
    cv.waitKey(0)
    cv.destroyAllWindows()

