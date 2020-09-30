"""
ENPM 673 - Perception for Autonomous robots
Group 18
Project 1
Virtual Cube Placement
"""

# Libraries imported
import numpy as np
import cv2
from numpy.linalg import norm


# Function to detect edges from grayscaled image
def alter_image(frame):
    img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscaling color image
    G_blur = cv2.GaussianBlur(img1, (7, 7), 0)  # applying gaussian blur with ksize=7
    edges = cv2.Canny(G_blur, 75, 200)  # using Canny edge detection algorithm
    return edges


# Function to return pure contour coordinates, area<2500 with just 4 bounding points
def clean_contours(contour_list):
    new_contour_list = []  # initialised list
    final_contour_list = []  # initialised list

    for contour in contour_list:
        if len(contour) == 4:  # if no. of contour points is 4
            new_contour_list.append(contour)  # gets appended to initial list
    for element in new_contour_list:  # refining list
        if cv2.contourArea(
                element) < 2500:  # if the contour area<2500, done to eliminate chance of detecting white paper background
            final_contour_list.append(element)  # appending in refined list
    return final_contour_list  # returning refined list


# Function to create contour edges list
def contour_create(frame):
    edges = alter_image(frame)
    contour_list = []
    final_contour_list = []
    ret, thresh = cv2.threshold(edges, 127, 255, 0)  # thresholded image
    h, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # getting contour points
    index = []
    for hier in hierarchy[0]:
        if hier[2] != -1:  # checking for the third element in hierarchy order. If it is not -1, it gets appended
            index.append(hier[2])  # if the value is -1, it is an unblunded point with no parents or child part
    for c in index:
        epsilon = 0.025 * cv2.arcLength(contours[c], True)
        approx = cv2.approxPolyDP(contours[c], epsilon, True)
        if len(
                approx) > 4:  # weeds out the inner most contour profile points, we need its parent only, the black square
            epsilon1 = 0.025 * cv2.arcLength(contours[c - 1], True)  # using coordinates for its parent, black square
            corners = cv2.approxPolyDP(contours[c - 1], epsilon1, True)
            contour_list.append(corners)
    final_contour_list = clean_contours(contour_list)
    return final_contour_list


# Function to draw cube
def cube(canvas, points):
    points = np.int32(points).reshape(-1, 2)
    for i, j in zip(range(4), range(4, 8)):
        canvas = cv2.drawContours(canvas, [points[:4]], -1, (0, 255, 0), 3)
        canvas = cv2.line(canvas, tuple(points[i]), tuple(points[j]), (0, 255, 0), 3)
        canvas = cv2.drawContours(canvas, [points[4:]], -1, (0, 255, 0), 3)
    return canvas


# Function to draw contour onto AR Tags
def draw_contours(f, cont):
    for i in cont:
        cv2.drawContours(f, [i], -1, (0, 255, 0), 2)
        cv2.imshow("Apri_Tag_Detection", f)


# Basic Function to return column
def column(matrix, i):
    return [row[i] for row in matrix]


# Function to rearrange matrix
def rearrange(column):
    matrix = np.zeros((4, 2))
    matrix[0] = column[3]
    matrix[1] = column[0]
    matrix[2] = column[1]
    matrix[3] = column[2]
    return matrix


# Function to rearrange p1 according to orientation
def rearrange_p1(p1):
    matrix = np.zeros((4, 2))
    matrix[0] = p1[0]
    matrix[1] = p1[1]
    matrix[2] = p1[2]
    matrix[3] = p1[3]
    return matrix


# To calculate homography matrix
def homography(point1, point2):
    x = []
    y = []
    x_bar = []
    y_bar = []
    A = []
    for i in range(len(point1)):
        x.append(point1[i][0])
        y.append(point1[i][1])
        x_bar.append(point2[i][0])
        y_bar.append(point2[i][1])
    for i in range(len(x)):
        A.append([x[i], y[i], 1, 0, 0, 0, -x_bar[i] * x[i], -x_bar[i] * y[i], -x_bar[i]])
        A.append([0, 0, 0, x[i], y[i], 1, -y_bar[i] * x[i], -y_bar[i] * y[i], -y_bar[i]])
    A = np.array(A)
    A.reshape(8, 9)
    [U, S, V] = np.linalg.svd(A)
    H = np.array(V[-1, :] / V[-1, 8])
    H = H.reshape(3, 3)
    return H


# Function to calculate Extrinsic values
def Extrinsic(H, K):
    H1 = np.linalg.inv(H)
    K_inverse = np.linalg.inv(K)
    B_new = K_inverse.dot(H1)
    B1_B = B_new[:, 0]
    B2_B = B_new[:, 1]
    B3_B = B_new[:, 2]
    R3_cross = np.cross(B1_B, B2_B)
    B1_B = B1_B.reshape(3, 1)
    B2_B = B2_B.reshape(3, 1)
    Lambda = 2 / (norm(K_inverse.dot(B1_B)) + norm(K_inverse.dot(B2_B)))
    r1 = Lambda * B1_B
    r2 = Lambda * B2_B
    r3 = (Lambda * Lambda * R3_cross)
    r3 = r3.reshape(3, 1)
    t = Lambda * B3_B
    R_Matrix = np.concatenate((r1, r2, r3), axis=1)
    return R_Matrix, t


# Runs all other functions(main)
def main_script(frame):
    global x1
    canvas = np.full(frame.shape, 0, dtype='uint8')  # creating empty canvas for masking purpose
    contour = contour_create(frame)
    cube_mat = []
    draw_contours(frame, contour)
    for i in range(len(contour)):
        extract_column = column(contour[i], 0)
        extract_column = rearrange(extract_column)
        H = homography(extract_column, rearrange_p1(x1))
        R_matrix, T = Extrinsic(H.copy(), K.copy())
        Connect, temp = cv2.projectPoints(cube_shape, R_matrix, T, K, np.zeros((1, 4)))
        cube_disp = cube(canvas, Connect)
        cube_mat.append(cube_disp)
    if cube_mat != []:
        for i in cube_mat:
            masking = cv2.add(canvas, i)
            canvas = masking
            display = cv2.add(frame, canvas)
            cv2.imshow('cubes drawn', display)  # displays shape of cube
            cv2.imshow('masking', masking)  # displays masking envt for better cube visualisation


# =============================================================================
# User input to accept choice
print(" \n")
print("Choose 1 for Tag0")
print("Choose 2 for Tag2")
print("Choose 3 for Multiple_tags")
print("")

selection = input("Enter your choice: ")
if int(selection) == 1:
    video = cv2.VideoCapture('Tag0.mp4')

elif int(selection) == 2:
    video = cv2.VideoCapture('Tag2.mp4')


elif int(selection) == 3:
    video = cv2.VideoCapture('multipleTags.mp4')
else:
    print("Invalid Selection")
    exit(0)

# =============================================================================

x1 = np.array([[0, 0], [199, 0], [199, 199], [0, 199]], dtype='float32')
K = np.array(
    [[1406.08415449821, 2.20679787308599, 1014.13643417416], [0, 1417.99930662800, 566.347754321696], [0, 0, 1]],
    dtype='float32')
cube_shape = np.array(
    [[0, 0, 0], [0, 200, 0], [200, 200, 0], [200, 0, 0], [0, 0, -200], [0, 200, -200], [200, 200, -200],
     [200, 0, -200]], dtype='float32')

while video.isOpened():
    opened, frame = video.read()

    if opened:
        a, b, c = frame.shape
        a = int(a / 2)
        b = int(b / 2)
        img = cv2.resize(frame, (b, a))
        main_script(img)
    else:
        break
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.waitKey(0)
cv2.destroyAllWindows()
video.release()