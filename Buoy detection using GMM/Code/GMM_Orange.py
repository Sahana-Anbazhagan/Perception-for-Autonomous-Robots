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

path = "Dataset1/Orange/Training/"


def calculateGaussian(x, mean, std):
    gauss = (1 / (std * math.sqrt(2 * math.pi))) * (math.exp(-((x - mean) ** 2) / (2 * std ** 2)))
    return gauss


def get_images():
    images = []
    for image in os.listdir(path):
        images.append(image)
    return images


def em_gmm(pixel):
    global n, mean1, mean2, mean3, std1, std2, std3

    while n != 40:
        prob1 = []
        prob2 = []
        prob3 = []

        b1 = []
        b2 = []
        b3 = []

        for pix in pixel:
            p1 = calculateGaussian(pix, mean1, std1)
            prob1.append(p1)
            p2 = calculateGaussian(pix, mean2, std2)
            prob2.append(p2)
            p3 = calculateGaussian(pix, mean3, std3)
            prob3.append(p3)
            b1.append((p1 * (1 / 3.0)) / (p1 * (1 / 3.0) + p2 * (1 / 3.0) + p3 * (1 / 3.0)))
            b2.append((p2 * (1 / 3.0)) / (p1 * (1 / 3.0) + p2 * (1 / 3.0) + p3 * (1 / 3.0)))
            b3.append((p3 * (1 / 3.0)) / (p1 * (1 / 3.0) + p2 * (1 / 3.0) + p3 * (1 / 3.0)))

        mean1 = np.sum(np.array(b1) * np.array(pixel)) / np.sum(np.array(b1))
        mean2 = np.sum(np.array(b2) * np.array(pixel)) / np.sum(np.array(b2))
        mean3 = np.sum(np.array(b3) * np.array(pixel)) / np.sum(np.array(b3))

        std1 = (np.sum(np.array(b1) * ((np.array(pixel)) - mean1) ** 2) / np.sum(np.array(b1))) ** (1 / 2.0)
        std2 = (np.sum(np.array(b2) * ((np.array(pixel)) - mean2) ** 2) / np.sum(np.array(b2))) ** (1 / 2.0)
        std3 = (np.sum(np.array(b3) * ((np.array(pixel)) - mean3) ** 2) / np.sum(np.array(b3))) ** (1 / 2.0)
        print("Iteration = ", n)
        print(mean1, mean2, mean3)
        print(std1, std2, std3)
        n = n + 1
    return mean1, mean2, mean3, std1, std2, std3


def append_all_pixel(images):
    pixel = []
    for image in images:
        image = cv2.imread("%s%s" % (path, image))
        image = image[:, :, 2]
        r, c = image.shape
        for j in range(0, r):
            for m in range(0, c):
                pix = image[j][m]
                pixel.append(pix)
    return pixel


def main():
    images = get_images()
    pixel = append_all_pixel(images)
    mean1, mean2, mean3, std1, std2, std3 = em_gmm(pixel)
    print("\n")
    print('final mean- ', mean1, mean2, mean3)
    print('final std- ', std1, std2, std3)


if __name__ == '__main__':
    main()
