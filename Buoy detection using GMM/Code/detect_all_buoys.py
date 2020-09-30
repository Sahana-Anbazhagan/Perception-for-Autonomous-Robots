import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

image_list=[]
x = list(range(0, 256))

o_mean_gaus1=np.array([238.14753766499953])
o_std_gaus1=np.array([8.417587391439978])
o_mean_gaus2=np.array([154.72662419580544])
o_std_gaus2=np.array([36.30795123353113])
o_mean_gaus3=np.array([252.24937318452012])
o_std_gaus3=np.array([2.346205081137928])

y_mean_gaus1 = np.array([230.99625753104914])
y_std_gaus1 = np.array([9.384343859959012])
y_mean_gaus2 = np.array([170.86092610188882])
y_std_gaus2 = np.array([38.29599853584383])
y_mean_gaus3 = np.array([235.26442800066383])
y_std_gaus3 = np.array([5.645941025982294])

g_mean_gaus1=np.array([184.90079776755584])
g_std_gaus1=np.array([20.577573078086086])
g_mean_gaus2=np.array([152.06534796946667])
g_std_gaus2=np.array([15.898153337093653])
g_mean_gaus3=np.array([236.81438517865615])
g_std_gaus3=np.array([11.714241624592178])

def gaussian(x, mu, sig):
    gauss = ((1 / (sig * math.sqrt(2 * math.pi))) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))
    return gauss


def yellow_plot_bellcurve():
    y_gauss_2 = gaussian(x, y_mean_gaus2, y_std_gaus2)
    y_gauss_1 = gaussian(x, y_mean_gaus1, y_std_gaus1)
    y_gauss_3 = gaussian(x, y_mean_gaus3, y_std_gaus3)
    plt.plot(y_gauss_2, 'b')
    plt.plot(y_gauss_1, 'g')
    plt.plot(y_gauss_3, 'r')
    plt.show()
    return y_gauss_2, y_gauss_1, y_gauss_3


def orange_plot_bellcurve():
    o_gauss_2 = gaussian(x, o_mean_gaus2, o_std_gaus2)
    o_gauss_1 = gaussian(x, o_mean_gaus1, o_std_gaus1)
    o_gauss_3 = gaussian(x, o_mean_gaus3, o_std_gaus3)
    plt.plot(o_gauss_1, 'b')
    plt.plot(o_gauss_2, 'g')
    plt.plot(o_gauss_3, 'r')
    plt.show()
    return o_gauss_2, o_gauss_1, o_gauss_3


def green_plot_bellcurve():
    g_gauss_1 = gaussian(x, g_mean_gaus1, g_std_gaus1)
    g_gauss_2 = gaussian(x, g_mean_gaus2, g_std_gaus2)
    g_gauss_3 = gaussian(x, g_mean_gaus3, g_std_gaus3)
    plt.plot(g_gauss_1, 'b')
    plt.plot(g_gauss_2, 'g')
    plt.plot(g_gauss_3, 'r')
    plt.show()
    return g_gauss_1, g_gauss_2, g_gauss_3


def image_process(frame, y_gauss_2, y_gauss_1, y_gauss_3, g_gauss_1, g_gauss_2, g_gauss_3, o_gauss_2, o_gauss_1,
                  o_gauss_3):
    frame_r = frame[:, :, 2]
    frame_g = frame[:, :, 1]
    frame_b = frame[:, :, 0]

    out1 = np.zeros(frame_g.shape, dtype=np.uint8)
    out3 = np.zeros(frame_r.shape, dtype = np.uint8)
    out2 = np.zeros(frame_r.shape, dtype = np.uint8)

    for i in range(0, frame_r.shape[0]):
        for j in range(0, frame_r.shape[1]):
            y = frame_r[i][j]
            z = frame_g[i][j]
            h = frame_r[i][j]

            if ((y_gauss_3[y] + y_gauss_3[z]) / 2) > 0.05 and ((y_gauss_2[y] + y_gauss_2[z]) / 2) < 0.015 and frame_b[i][j] < 130:
                out2[i][j] = 255
            else:
                out2[i][j] = 0

            if g_gauss_3[z] > 0.025 and g_gauss_2[z] < 0.025 and g_gauss_1[z] < 0.02 and frame_r[i][j] < 180:
                out1[i][j] = 255
            else:
                out1[i][j] = 0

            if o_gauss_3[h] > 0.050 and frame_b[i][j] < 150:
                out3[i][j] = 255
            if o_gauss_1[h] > 0.05 and frame_b[i][j] < 150:
                out3[i][j] = 0
            if o_gauss_2[h] > 0.01 and frame_b[i][j] < 150:
                out3[i][j] = 0

    ret, threshold2 = cv2.threshold(out2, 240, 255, cv2.THRESH_BINARY)
    kernel2 = np.ones((3, 3), np.uint8)

    dilation2 = cv2.dilate(threshold2, kernel2, iterations=6)
    contours2, _ = cv2.findContours(dilation2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    ret, threshold3 = cv2.threshold(out1, 240, 255, cv2.THRESH_BINARY)
    kernel3 = np.ones((2, 2), np.uint8)

    dilation3 = cv2.dilate(threshold3, kernel3, iterations=9)
    contours1, _ = cv2.findContours(dilation3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    kernel1 = np.ones((2, 2), np.uint8)
    dilation1 = cv2.dilate(out3, kernel1, iterations=6)
    contours3, _ = cv2.findContours(dilation1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame = draw_circle(frame, contours1, contours2, contours3)

    cv2.imshow('buoy detection', frame)
    image_list.append(frame)
    return


def draw_circle(frame, contours1, contours2, contours3):
    for contour in contours2:
        if cv2.contourArea(contour) > 20:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y) - 1)
            radius = int(radius) - 1
            if radius > 12:
                cv2.circle(frame, center, radius, (0, 255, 255), 2)

    for contour in contours1:
        if cv2.contourArea(contour) > 31:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if 13 < radius < 16:
                cv2.circle(frame, center, radius, (0, 255, 0), 2)

    for contour in contours3:
        if cv2.contourArea(contour) > 18:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if radius > 13:
                cv2.circle(frame, center, radius, (0, 0, 255), 2)
    return frame


def main():
    y_gauss_2, y_gauss_1, y_gauss_3 = yellow_plot_bellcurve()
    g_gauss_1, g_gauss_2, g_gauss_3 = green_plot_bellcurve()
    o_gauss_2, o_gauss_1, o_gauss_3 = orange_plot_bellcurve()

    video = cv2.VideoCapture("detectbuoy.avi")
    while video.isOpened():
        opened, frame = video.read()
        if opened:
            image_process(frame, y_gauss_2, y_gauss_1, y_gauss_3, g_gauss_1, g_gauss_2, g_gauss_3, o_gauss_2, o_gauss_1,
                          o_gauss_3)
            cv2.waitKey(1)
        else:
            break
        k = cv2.waitKey(15) & 0xff
        if k == 27:
            break
    
    out = cv2.VideoWriter('Buoy_detect.avi', cv2.VideoWriter_fourcc(*'XVID'), 5.0, (640, 480))
    for frame in image_list:
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    main()
