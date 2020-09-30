import cv2
import numpy as np
import copy
import sys


def alter_image(frame):
    img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    G_blur = cv2.medianBlur(img1, ksize = 5)
    edges = cv2.Canny(G_blur, 75, 200)
    return edges


def clean_contours(contour_list):
    new_contour_list = []
    final_contour_list = []

    for contour in contour_list:
        if len(contour) == 4:
            new_contour_list.append(contour)

    for element in new_contour_list:
        if cv2.contourArea(element) < 2500:
            final_contour_list.append(element)

    return final_contour_list


def contour_create(frame):
    edges = alter_image(frame)
    contour_list = list()
    final_contour_list = list()

    ret, thresh = cv2.threshold(edges, 127, 255, 0)
    p, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    index = list()
    for hier in hierarchy[0]:
        if hier[3] != -1:
            index.append(hier[3])

    # loop over the contours
    for c in index:

        epsilon = 0.025 * cv2.arcLength(contours[c], True)
        approx = cv2.approxPolyDP(contours[c], epsilon, True)

        if len(approx) > 4:
            epsilon1 = 0.025 * cv2.arcLength(contours[c - 1], True)
            corners = cv2.approxPolyDP(contours[c - 1], epsilon1, True)
            contour_list.append(corners)
    final_contour_list = clean_contours(contour_list)

    return final_contour_list


def draw_contours(f, cont):
    for i in cont:
        cv2.drawContours(f, [i], -1, (0, 255, 0), 2)
        cv2.imshow("AR_Tag_Detection", f)



def column(matrix, i):
    return [row[i] for row in matrix]


# main function to process the tag
def rearrange(column):
    matrix = np.zeros((4,2))
    matrix[0] = column[3]
    matrix[1] = column[0]
    matrix[2] = column[1]
    matrix[3] = column[2]
    return matrix

def rearrange_p1(p1):
    matrix = np.zeros((4,2))
    matrix[0] = p1[0]
    matrix[1] = p1[1]
    matrix[2] = p1[2]
    matrix[3] = p1[3]
    return matrix


def homography(point1, point2):
    x = []
    y = []
    x_bar = []
    y_bar = []
    H = []
    for i in range(len(point1)):
        x.append(point1[i][0])
        y.append(point1[i][1])
        x_bar.append(point2[i][0])
        y_bar.append(point2[i][1])
    for i in range(len(x)):
        H.append([x[i], y[i], 1, 0, 0, 0, -x_bar[i] * x[i], -x_bar[i] * y[i], -x_bar[i]])
        H.append([0, 0, 0, x[i], y[i], 1, -y_bar[i] * x[i], -y_bar[i] * y[i], -y_bar[i]])
    H = np.array(H)
    H.reshape(8, 9)
    [U, S, V] = np.linalg.svd(H)
    H = np.array(V[-1, :])
    H = H.reshape(3,3)
    return H

def orientation_of_tag(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # ret, th1 = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    white_region = 255
    data_region = th1[50:150, 50:150]
    cv2.imshow("data",data_region)

    if data_region[90, 90] == white_region:
        position = "botton_right"
    elif data_region[10, 90] == white_region:
        position = "top_right"
    elif data_region[90, 10] == white_region:
        position = "bottom_left"
    elif data_region[10, 10] == white_region:
        position = "top_left"
    else:
        return None, None
    return (data_region, position)

def decode(data_region, orientation):
    white = 255
    data = []
    new_data = []

    if orientation != None:
        if data_region[25, 25] == white:
            data.append(1)
        else:
            data.append(0)

        if data_region[25, 75] == white:
            data.append(1)
        else:
            data.append(0)

        if data_region[75, 75] == white:
            data.append(1)
        else:
            data.append(0)

        if data_region[75, 25] == white:
            data.append(1)
        else:
            data.append(0)

        if orientation == "bottom_right":
            # print("bottom right")
            new_data.append(data[0])
            new_data.append(data[1])
            new_data.append(data[2])
            new_data.append(data[3])

        if orientation == "bottom_left":
            # print("bottom left")
            new_data.append(data[1])
            new_data.append(data[2])
            new_data.append(data[3])
            new_data.append(data[0])

        if orientation == "top_left":
            # print("top_left")
            new_data.append(data[2])
            new_data.append(data[3])
            new_data.append(data[0])
            new_data.append(data[1])

        if orientation == "top_right":
            # print("top right")
            new_data.append(data[3])
            new_data.append(data[0])
            new_data.append(data[1])
            new_data.append(data[2])

        if not new_data == []:
            print("Encoded Data = ")
            print(new_data)

    return new_data



def warp(frame, H):
    H_matrix_inverse = np.linalg.inv(H)
    tag = np.zeros((200, 200, 3))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, grayImage = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    q,w = grayImage.shape
    for v in range(200):
        for u in range(200):
            x, y, z = np.matmul(H_matrix_inverse, [v, u, 1])
            if y/z != float('inf') and (x / z != float('inf')):
                if (int(q) > int(y / z) > 0) and (int(w) > int(x / z) > 0) :
                    tag[v][u][0] = grayImage[int(y / z)][int(x / z)]
                    tag[v][u][1] = grayImage[int(y / z)][int(x / z)]
                    tag[v][u][2] = grayImage[int(y / z)][int(x / z)]
    tag = tag.astype('uint8')
    return tag


def main_script(frame):
    global x1
    contour = contour_create(frame)
    print("Corners of the TAG = ", contour)
    print("\n")
    draw_contours(frame, contour)
    for i in range(len(contour)):
        extract_column = column(contour[i], 0)
        extract_column = rearrange(extract_column)
        H = homography(extract_column, rearrange_p1(x1))

        # tag = cv2.warpPerspective(frame, H, (200, 200))
        tag = warp(frame, H)
        cv2.imshow("Tag after perspective transformation", tag)
        data_region, orientation = orientation_of_tag(tag)
        encoded_data = decode(data_region, orientation)


print(" \n")
print("Choose 1 for Tag0")
print("Choose 2 for Tag2")
print("Choose 3 for Multiple_tags")
print("")

selection = input("Enter your choice: ")
if int(selection) == 1:
    video = cv2.VideoCapture('Tag0.mp4')
    bbox = (405, 180, 320, 247)
    flag = 0

elif int(selection) == 2:
    video = cv2.VideoCapture('Tag2.mp4')
    bbox = (268, 149, 441, 382)
    flag = 0

elif int(selection) == 3:
    video = cv2.VideoCapture('multipleTags.mp4')
    flag = 1
else:
    print("Invalid Selection")
    exit(0)


lena = cv2.imread('Lena.png')
lena = cv2.resize(lena, (200, 200))
x1 = [[0, 0], [199, 0], [199, 199], [0, 199]]

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[2]

if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
if tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
if tracker_type == 'MOSSE':
    tracker = cv2.TrackerMOSSE_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()



if not flag == 1:

    ok, f = video.read()
    frame = cv2.resize(f, (0, 0), fx=0.5, fy=0.5)

    ok = tracker.init(frame, bbox)


while video.isOpened():
    opened, frame = video.read()

    if flag == 1:
        a, b, c = frame.shape
        a = a / 2
        b = b / 2
        frame = cv2.resize(frame, (b, a))
        main_script(frame)

    elif opened and not (flag == 1):
        a, b, c = frame.shape
        a = int(a / 2)
        b = int(b / 2)
        frame = cv2.resize(frame, (b, a))
        ok, bbox = tracker.update(frame)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        cv2.imshow("frame",frame)

        roi = frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
        main_script(roi)
    else:
        break
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break
cv2.destroyAllWindows()
video.release()