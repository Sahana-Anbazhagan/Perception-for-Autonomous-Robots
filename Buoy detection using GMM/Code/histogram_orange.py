import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math

path = "Dataset1/Orange/Training/"


def gaussian(x, mu, sig):
    gauss = ((1 / (sig * math.sqrt(2 * math.pi))) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))
    return gauss


def get_images():
    images = []
    for image in os.listdir(path):
        images.append(image)
    return images


def calc_hist(images):
    histogram_b = np.zeros((256, 1))
    histogram_g = np.zeros((256, 1))
    histogram_r = np.zeros((256, 1))

    for image in images:
        imagee = cv2.imread("%s%s" % (path, image))
        img = cv2.GaussianBlur(imagee, (5, 5), 0)
        color = ("b", "g", "r")
        for i, col in enumerate(color):
            if col == 'b':
                histr_b = cv2.calcHist([img], [i], None, [256], [0, 256])
                histogram_b = np.column_stack((histogram_b, histr_b))
            if col == 'g':
                histr_g = cv2.calcHist([img], [i], None, [256], [0, 256])
                histogram_g = np.column_stack((histogram_g, histr_g))
            if col == 'r':
                histr_r = cv2.calcHist([img], [i], None, [256], [0, 256])
                histogram_r = np.column_stack((histogram_r, histr_r))
    return img, histogram_r, histogram_g, histogram_b


def histogram_avg(histogram_r, histogram_g, histogram_b):
    histogram_avg_r = np.sum(histogram_r, axis=1) / (histogram_r.shape[1] - 1)
    histogram_avg_g = np.sum(histogram_g, axis=1) / (histogram_g.shape[1] - 1)
    histogram_avg_b = np.sum(histogram_b, axis=1) / (histogram_b.shape[1] - 1)
    plt.plot(histogram_avg_r, color='r')
    plt.plot(histogram_avg_g, color='g')
    plt.plot(histogram_avg_b, color='b')
    plt.show()


def plot_bellcurve(img):
    (mean, stds) = cv2.meanStdDev(img)
    x = list(range(0, 255))
    mean = mean[2]
    std = stds[2]
    gauss = gaussian(x, mean, std)

    plt.plot(gauss, color='orange')
    plt.show()


def main():
    images = get_images()
    img, histogram_r, histogram_g, histogram_b = calc_hist(images)
    histogram_avg( histogram_r, histogram_g, histogram_b)
    plot_bellcurve(img)


if __name__ == '__main__':
    main()
