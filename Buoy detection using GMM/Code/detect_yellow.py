import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


x = list(range(0, 256))

mean_gaus1 = np.array([230.99625753104914])
std_gaus1 = np.array([9.384343859959012])
mean_gaus2 = np.array([170.86092610188882])
std_gaus2 = np.array([38.29599853584383])
mean_gaus3 = np.array([235.26442800066383])
std_gaus3 = np.array([5.645941025982294])

def gaussian(x, mu, sig):
    gauss = ((1/(sig*math.sqrt(2*math.pi)))*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))
    return gauss


def plot_bellcurve():
    gauss_2 = gaussian(x, mean_gaus2, std_gaus2)
    gauss_1 = gaussian(x, mean_gaus1, std_gaus1)
    gauss_3 = gaussian(x, mean_gaus3, std_gaus3)
    plt.plot(gauss_2, 'b')
    plt.plot(gauss_1, 'g')
    plt.plot(gauss_3, 'r')
    plt.show()
    return gauss_2, gauss_1, gauss_3


def image_process(frame, gauss_2, gauss_1, gauss_3):
    frame_r=frame[:,:,2]
    frame_g = frame[:,:,1]
    frame_b=frame[:,:,0]

    out=np.zeros(frame_r.shape, dtype = np.uint8)

    for i in range(0, frame_r.shape[0]):
        for j in range(0, frame_r.shape[1]):
            y = frame_r[i][j]
            z = frame_g[i][j]
            if ((gauss_3[y] + gauss_3[z]) / 2) > 0.045 and ((gauss_2[y] + gauss_2[z]) / 2) < 0.01 and frame_b[i][j] < 130:
                out[i][j] = 255
            else:
                out[i][j] = 0

    ret, threshold2 = cv2.threshold(out, 240, 255, cv2.THRESH_BINARY)
    kernel2 = np.ones((3, 3), np.uint8)

    dilation2 = cv2.dilate(threshold2, kernel2, iterations=6)
    contours2, _ = cv2.findContours(dilation2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame = draw_circle(frame, contours2)
    cv2.imshow('yellow detection', frame)
    return


def draw_circle(frame, contours1):
    for contour in contours1:
        if cv2.contourArea(contour) > 20:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y) - 1)
            radius = int(radius) - 1
            if radius > 12:
                cv2.circle(frame, center, radius, (0, 255, 255), 2)
    return frame


def main():
    gauss_2, gauss_1, gauss_3 = plot_bellcurve()
    video = cv2.VideoCapture("detectbuoy.avi")
    while video.isOpened():
        opened, frame = video.read()
        if opened:
            image_process(frame, gauss_2, gauss_1, gauss_3)
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
