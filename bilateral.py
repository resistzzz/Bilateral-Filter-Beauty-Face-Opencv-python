import cv2 as cv
import numpy as np


def bilateralFilter(img, K, sigmas):
    m, n = img.shape
    img_n = np.zeros((m+2*K, n+2*K))

    # padding
    img_n[K:m+K, K:n+K] = img
    img_n[0:K, K:n+K] = img[0:K, 0:n]   # up
    img_n[m+K: m+2*K, K:n+K] = img[m-K:m, 0:n]     # dowm
    img_n[K:m+K, 0:K] = img[0:m, 0:K]   # left
    img_n[K:m+K, n+K:n+2*K] = img[0:m, n-K:n]     #right

    sigma_d = sigmas[0]
    sigma_r = sigmas[1]

    x = np.linspace(-K, K, 2*K+1)
    x, y = np.meshgrid(x, x)
    w1 = 1.0/(2*np.pi*sigma_d**2)*np.exp(-(x**2+y**2)/(2*sigma_d**2))

    for i in range(K, m+K):
        for j in range(K, n+K):
            w2 = 1.0/(2*np.pi*sigma_r**2)*np.exp(-(img_n[i-K:i+K+1, j-K:j+K+1] - img_n[i, j])**2/(2*sigma_r**2))
            w = w1 * w2
            s = img_n[i-K:i+K+1, j-K:j+K+1] * w
            img_n[i, j] = np.sum(s)/np.sum(w)

    img_out = img_n[K:m+K, K:n+K]
    img_out = img_out.astype(np.uint8)
    return img_out

if __name__ == '__main__':

    fname = './pictures/stone.png'
    sigma_s = 20
    sigma_r = 30
    img = cv.imread(fname, 0)
    outImg = bilateralFilter(img, 10, (sigma_s, sigma_r))
    outImg_opencv = cv.bilateralFilter(img, 10, sigma_r, sigma_s)
    # difference = np.sum(outImg_opencv!=outImg)
    cv.imshow('img', img)
    cv.imshow('outImg', outImg)
    cv.imshow('opencv', outImg_opencv)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite('./result/outImg_own.png', outImg)
    cv.imwrite('./result/outImg_opencv.png', outImg_opencv)



