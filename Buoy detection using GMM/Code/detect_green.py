import cv2
import numpy as np
from matplotlib import pyplot as plt
import math


x=list(range(0, 256))

mean_gaus1=np.array([184.90079776755584])
std_gaus1=np.array([20.577573078086086])
mean_gaus2=np.array([152.06534796946667])
std_gaus2=np.array([15.898153337093653])
mean_gaus3=np.array([236.81438517865615])
std_gaus3=np.array([11.714241624592178])

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
    frame_g=frame[:,:,1]
    frame_r=frame[:,:,2]
    out=np.zeros(frame_g.shape, dtype = np.uint8)

    for i in range(0, frame_g.shape[0]):
        for j in range(0, frame_g.shape[1]):
            z = frame_g[i][j]

            if gauss_3[z] > 0.025 and gauss_2[z] < 0.025 and gauss_1[z] < 0.02 and frame_r[i][j] < 180:
                #                     print(z)
                out[i][j] = 255
            else:
                out[i][j] = 0
    ret, threshold3 = cv2.threshold(out, 240, 255, cv2.THRESH_BINARY)
    kernel3 = np.ones((2, 2), np.uint8)
    
    dilation3 = cv2.dilate(threshold3, kernel3, iterations=9) 
    contours1, ret = cv2.findContours(dilation3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame = draw_circle(frame, contours1)
    cv2.imshow('Green detection', frame)
    return


def draw_circle(frame, contours1):
    for contour in contours1:
        if cv2.contourArea(contour) > 31:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if 13 < radius < 16:
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
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
