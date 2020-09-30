import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


x=list(range(0, 256))

mean_gaus1=np.array([238.14753766499953])
std_gaus1=np.array([8.417587391439978])
mean_gaus2=np.array([154.72662419580544])
std_gaus2=np.array([36.30795123353113])
mean_gaus3=np.array([252.24937318452012])
std_gaus3=np.array([2.346205081137928])

def gaussian(x, mu, sig):
    gauss = ((1/(sig*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))
    return gauss


def plot_bellcurve():
    gauss_1 = gaussian(x, mean_gaus1, std_gaus1)
    gauss_2 = gaussian(x, mean_gaus2, std_gaus2)
    gauss_3 = gaussian(x, mean_gaus3, std_gaus3)
    plt.plot(gauss_1, 'b')
    plt.plot(gauss_2, 'g')
    plt.plot(gauss_3, 'r')
    plt.show()
    return gauss_1, gauss_2, gauss_3


def image_process(frame, gauss_1, gauss_2, gauss_3):
    frame_r = frame[:,:,2]
    frame_b = frame[:,:,0]
    out = np.zeros(frame_r.shape, dtype = np.uint8)

    for i in range(0, frame_r.shape[0]):
        for j in range(0, frame_r.shape[1]):
            y = frame_r[i][j]

            if gauss_1[y] > 0.05 and frame_b[i][j] < 150:
                out[i][j] = 0

            if gauss_2[y] > 0.01 and frame_b[i][j] < 150:
                out[i][j] = 0

            if gauss_3[y] > 0.050 and frame_b[i][j] < 150:
                out[i][j] = 255

    kernel1 = np.ones((2, 2), np.uint8)

    dilation1 = cv2.dilate(out, kernel1, iterations=6)

    contours1, _ = cv2.findContours(dilation1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame = draw_circle(frame, contours1)
    cv2.imshow('Orange detection', frame)
    return


def draw_circle(frame, contours1):
    for contour in contours1:
        if cv2.contourArea(contour) > 20:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if radius > 13:
                cv2.circle(frame, center, radius, (0, 0, 255), 2)
    return frame


def main():
    gauss_1, gauss_2, gauss_3 = plot_bellcurve()
    video = cv2.VideoCapture("detectbuoy.avi")
    while video.isOpened():
        opened, frame = video.read()
        if opened:
            image_process(frame, gauss_1, gauss_2, gauss_3)
            cv2.waitKey(1)
        else:
            break
        k = cv2.waitKey(15) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    main()
