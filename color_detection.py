import cv2
import numpy as np


def get_filtered_by_colors_image(img, lower: np.ndarray, upper: np.ndarray):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(imgHSV, lower, upper)
    img_result = cv2.bitwise_and(img, img, mask=mask)
    return img_result


def increase_contrast(img):
    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # cv2.imshow("lab", lab)
    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # cv2.imshow('l_channel', l)
    # cv2.imshow('a_channel', a)
    # cv2.imshow('b_channel', b)
    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # cv2.imshow('CLAHE output', cl)
    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))
    # cv2.imshow('limg', limg)
    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    # cv2.imshow('final', final)

    return final


def rough_image_hsv(img_hsv):
    max_values = [179, 255, 255]
    for x in range(img_hsv.shape[1]):
        for y in range(img_hsv.shape[0]):
            h, s, v = img_hsv[y, x]

            h = (h // 20) * 20
            s = (h // 20) * 20
            v = (h // 20) * 20

            img_hsv[y, x][0] = h
            img_hsv[y, x][1] = s
            img_hsv[y, x][2] = v
    return img_hsv


def empty(a):
    pass


def detection_tools_rgb():
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    cv2.createTrackbar("R Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("R Max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("G Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("G Max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("B Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("B Max", "TrackBars", 255, 255, empty)

    img = cv2.imread("example2.jpg")
    img = img[:, int(img.shape[1]*0.9):]

    img = cv2.resize(img, (320, 640))

    lower = np.array([85, 15, 10])
    upper = np.array([122, 235, 255])

    while True:
        h_min = cv2.getTrackbarPos("R Min", "TrackBars")
        h_max = cv2.getTrackbarPos("R Max", "TrackBars")
        s_min = cv2.getTrackbarPos("G Min", "TrackBars")
        s_max = cv2.getTrackbarPos("G Max", "TrackBars")
        v_min = cv2.getTrackbarPos("B Min", "TrackBars")
        v_max = cv2.getTrackbarPos("B Max", "TrackBars")
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(img, lower, upper)
        img_result = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow("aaaa", img_result)
        cv2.waitKey(1)
        if cv2.waitKey(1) == 27:# & 0xFF == ord('q'):
            break
    print(f"lower = [{lower[0]}, {lower[1]}, {lower[2]}]")
    print(f"upper = [{upper[0]}, {upper[1]}, {upper[2]}]")


def detection_tools_hsv():
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
    cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

    img = cv2.imread("example2.jpg")
    img = img[:, int(img.shape[1]*0.9):]

    img = cv2.resize(img, (320, 640))
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower = np.array([85, 15, 10])
    upper = np.array([122, 235, 255])

    while True:
        h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
        h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
        s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
        s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
        v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
        v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHSV, lower, upper)
        imgResult = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow("aaaa", imgResult)
        cv2.waitKey(1)
        if cv2.waitKey(1) == 27:# & 0xFF == ord('q'):
            break
    print(f"lower = [{lower[0]}, {lower[1]}, {lower[2]}]")
    print(f"upper = [{upper[0]}, {upper[1]}, {upper[2]}]")


if __name__ == '__main__':
    detection_tools_hsv()
# lower = [0, 0, 0]
# upper = [179, 86, 115]
