# Bolt Tracking
import cv2
import numpy as np
import glob


def get_ROI():
    ROI = [[[269, 75], [303, 139]], [[243, 67], [277, 131]], [[255, 73], [289, 137]], [[270, 77], [304, 141]],
           [[276, 75], [310, 139]], [[302, 83], [336, 147]], [[338, 95], [372, 159]]]
    return ROI


def get_template(Image, frame_number):
    ROI = get_ROI()
    if frame_number == 0:
        ROI = ROI[0]
    if frame_number == 29:
        ROI = ROI[1]
    if frame_number == 104:
        ROI = ROI[2]
    if frame_number == 125:
        ROI = ROI[3]
    if frame_number == 150:
        ROI = ROI[4]
    if frame_number == 171:
        ROI = ROI[5]
    if frame_number == 220:
        ROI = ROI[6]
    x = ROI[0][1]
    y = ROI[0][0]
    xn = ROI[1][1]
    yn = ROI[1][0]
    Temp = Image[x:xn, y:yn]
    Temp = cv2.equalizeHist(Temp)
    return ROI, Temp


def update_p(p, Delta_p):
    denominator = (1 + Delta_p[0]) * (1 + Delta_p[3]) - Delta_p[1] * Delta_p[2]
    Delta_p_0 = (-Delta_p[0] - Delta_p[0] * Delta_p[3] + Delta_p[1] * Delta_p[2]) / denominator
    Delta_p_1 = (-Delta_p[1]) / denominator
    Delta_p_2 = (-Delta_p[2]) / denominator
    Delta_p_3 = (-Delta_p[3] - Delta_p[0] * Delta_p[3] + Delta_p[1] * Delta_p[2]) / denominator
    Delta_p_4 = (-Delta_p[4] - Delta_p[3] * Delta_p[4] + Delta_p[2] * Delta_p[5]) / denominator
    Delta_p_5 = (-Delta_p[5] - Delta_p[0] * Delta_p[5] + Delta_p[1] * Delta_p[4]) / denominator

    p[0] += Delta_p_0 + p[0] * Delta_p_0 + p[2] * Delta_p_1
    p[1] += Delta_p_1 + p[1] * Delta_p_0 + p[3] * Delta_p_1
    p[2] += Delta_p_2 + p[0] * Delta_p_2 + p[2] * Delta_p_3
    p[3] += Delta_p_3 + p[1] * Delta_p_2 + p[3] * Delta_p_3
    p[4] += Delta_p_4 + p[0] * Delta_p_4 + p[2] * Delta_p_5
    p[5] += Delta_p_5 + p[1] * Delta_p_4 + p[3] * Delta_p_5

    return p


def Affine_LK_TRacker(img, tmp, rect, p):
    # Initialization
    rows, cols = tmp.shape

    iteration = 200
    # Calculate gradient of template
    grad_x = cv2.Sobel(tmp, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(tmp, cv2.CV_64F, 0, 1, ksize=5)

    # Calculate Jacobian
    x = np.array(range(cols))
    y = np.array(range(rows))
    x, y = np.meshgrid(x, y)
    ones = np.ones((rows, cols))
    zeros = np.zeros((rows, cols))
    row1 = np.stack((x, zeros, y, zeros, ones, zeros), axis=2)
    row2 = np.stack((zeros, x, zeros, y, zeros, ones), axis=2)
    jacob = np.stack((row1, row2), axis=2)

    # Set gradient of a pixel into 1 by 2 vector
    grad = np.stack((grad_x, grad_y), axis=2)
    grad = np.expand_dims((grad), axis=2)
    steepest_descents = np.matmul(grad, jacob)
    steepest_descents_trans = np.transpose(steepest_descents, (0, 1, 3, 2))

    # Compute Hessian matrix
    H = np.matmul(steepest_descents_trans, steepest_descents).sum((0, 1))

    for i in range(iteration):
        # Calculate warp image
        warp_mat = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
        warp_img = cv2.warpAffine(img, warp_mat, (0, 0))
        warp_img = warp_img[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        # Equalize the image
        warp_img = cv2.equalizeHist(warp_img)

        # Compute the error term
        error = tmp.astype(float) - warp_img.astype(float)

        # Compute steepest-gradient-descent update
        error = error.reshape((rows, cols, 1, 1))
        update = (steepest_descents_trans * error).sum((0, 1))
        Delta_p = np.matmul(np.linalg.pinv(H), update).reshape((-1))

        # Update p
        p = update_p(p, Delta_p)
    return p


def recalculate_coordinates(ROI, warp_mat):
    ROI_other = [[ROI[1][0], ROI[0][1]], [ROI[0][0], ROI[1][1]]]
    newROI = np.hstack((ROI, [[1], [1]]))
    newROI_other = np.hstack((ROI_other, [[1], [1]]))
    newROI = np.dot(warp_mat, newROI.T).astype(np.int32)
    newROI_other = np.dot(warp_mat, newROI_other.T).astype(np.int32)
    pts = np.array([newROI.T[0], newROI_other.T[0], newROI.T[1], newROI_other.T[1]])
    pts = pts.reshape((-1, 1, 2))
    return pts


def main():
    frame_increase = 0
    image_list = []
    image1 = []
    ROI = get_ROI()
    for img in glob.glob((r"Bolt2/img/*.jpg")):
        image1.append(img)
    image1.sort()
    for img in image1:
        image = cv2.imread(img)
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if frame_increase == 0 or frame_increase == 29 or frame_increase == 104 or frame_increase == 125 or frame_increase == 150 or frame_increase == 171 or frame_increase == 220:
            p = np.zeros(6)
            ROI, template = get_template(grey, frame_increase)
            # cv2.imshow('template', template)

        # Using Lk to track
        p = Affine_LK_TRacker(grey, template, ROI, p)

        # Update new W matrix
        warp_mat = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
        warp_mat = cv2.invertAffineTransform(warp_mat)

        pts = recalculate_coordinates(ROI, warp_mat)

        cv2.polylines(image, [pts], True, (0, 255, 0), 2)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 360, 240)
        cv2.imshow("image", image)
        image_list.append(image)
        cv2.waitKey(1)

        frame_increase += 1
        # print(frame_increase)

    height, width, layers = image_list[0].shape
    size = (width, height)


    out = cv2.VideoWriter('bolt_tracking.avi',cv2.VideoWriter_fourcc(*'DIVX'), 5.0, size)

    for i in range(len(image_list)):
        # writing to a image array
        out.write(image_list[i])
    out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
