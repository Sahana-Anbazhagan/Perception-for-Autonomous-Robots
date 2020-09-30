import cv2
import numpy as np
import os
import math

n = 0
mean1 = 190
mean2 = 150
mean3 = 250
std1 = 10

std2 = 10
std3 = 10

path = "Dataset1/Yellow/Training/"


def calculateGaussian(x, mean, std):
    gauss = (1 / (std * math.sqrt(2 * math.pi))) * (math.exp(-((x - mean) ** 2) / (2 * std ** 2)))
    return gauss


def get_images():
    images = []
    for image in os.listdir(path):
        images.append(image)
    return images


def em_gmm(pixel1, pixel2):
    global n, mean1, mean2, mean3, std1, std2, std3

    while n != 50:
        b11 = []
        b12 = []
        b13 = []
        b21 = []
        b22 = []
        b23 = []
        for pix1 in pixel1:
            p11 = calculateGaussian(pix1, mean1, std1)
            p12 = calculateGaussian(pix1, mean2, std2)
            p13 = calculateGaussian(pix1, mean3, std3)
            b11.append((p11 * (1 / 3.0)) / (p11 * (1 / 3.0) + p12 * (1 / 3.0) + p13 * (1 / 3.0)))
            b12.append((p12 * (1 / 3.0)) / (p11 * (1 / 3.0) + p12 * (1 / 3.0) + p13 * (1 / 3.0)))
            b13.append((p13 * (1 / 3.0)) / (p11 * (1 / 3.0) + p12 * (1 / 3.0) + p13 * (1 / 3.0)))
        for pix2 in pixel2:
            p21 = calculateGaussian(pix2, mean1, std1)
            p22 = calculateGaussian(pix2, mean2, std2)
            p23 = calculateGaussian(pix2, mean3, std3)
            b21.append((p21 * (1 / 3.0)) / (p21 * (1 / 3.0) + p22 * (1 / 3.0) + p23 * (1 / 3.0)))
            b22.append((p22 * (1 / 3.0)) / (p21 * (1 / 3.0) + p22 * (1 / 3.0) + p23 * (1 / 3.0)))
            b23.append((p23 * (1 / 3.0)) / (p21 * (1 / 3.0) + p22 * (1 / 3.0) + p23 * (1 / 3.0)))
            
        m11 = np.sum(np.array(b11) * np.array(pixel1)) / np.sum(np.array(b11))
        m12 = np.sum(np.array(b12) * np.array(pixel1)) / np.sum(np.array(b12))
        m13 = np.sum(np.array(b13) * np.array(pixel1)) / np.sum(np.array(b13))
        s11 = (np.sum(np.array(b11) * ((np.array(pixel1)) - mean1) ** 2) / np.sum(np.array(b11))) ** (1 / 2.0)
        s12 = (np.sum(np.array(b12) * ((np.array(pixel1)) - mean2) ** 2) / np.sum(np.array(b12))) ** (1 / 2.0)
        s13 = (np.sum(np.array(b13) * ((np.array(pixel1)) - mean3) ** 2) / np.sum(np.array(b13))) ** (1 / 2.0)
        m21 = np.sum(np.array(b21) * np.array(pixel2)) / np.sum(np.array(b21))
        m22 = np.sum(np.array(b22) * np.array(pixel2)) / np.sum(np.array(b22))
        m23 = np.sum(np.array(b23) * np.array(pixel2)) / np.sum(np.array(b23))
        s21 = (np.sum(np.array(b21) * ((np.array(pixel2)) - mean1) ** 2) / np.sum(np.array(b21))) ** (1 / 2.0)
        s22 = (np.sum(np.array(b22) * ((np.array(pixel2)) - mean2) ** 2) / np.sum(np.array(b22))) ** (1 / 2.0)
        s23 = (np.sum(np.array(b23) * ((np.array(pixel2)) - mean3) ** 2) / np.sum(np.array(b23))) ** (1 / 2.0)
        n = n + 1
        mean1 = (m11 + m21) / 2
        mean2 = (m12 + m22) / 2
        mean3 = (m13 + m23) / 2
        std1 = (s11 + s21) / 2
        std2 = (s12 + s22) / 2
        std3 = (s13 + s23) / 2
        print("Iteration = ", n)
        print(mean1, mean2, mean3)
        print(std1, std2, std3)
        n = n + 1
    return mean1, mean2, mean3, std1, std2, std3


def append_all_pixel(images):
    pixel1 = []
    pixel2 = []
    for image in images:
        image = cv2.imread("%s%s" % (path, image))

        image1 = (image[:, :, 1])
        r, c = image1.shape
        for j in range(0, r):
            for m in range(0, c):
                pix1 = image1[j][m]
                pixel1.append(pix1)
        image2 = (image[:, :, 2])
        r, c = image2.shape
        for j in range(0, r):
            for m in range(0, c):
                pix2 = image2[j][m]
                pixel2.append(pix2)
    return pixel1, pixel2


def main():
    images = get_images()
    pixel1, pixel2 = append_all_pixel(images)
    mean1, mean2, mean3, std1, std2, std3 = em_gmm(pixel1, pixel2)
    print("\n")
    print('final mean- ', mean1, mean2, mean3)
    print('final std- ', std1, std2, std3)


if __name__ == '__main__':
    main()
