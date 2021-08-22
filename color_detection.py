import cv2
import numpy as np


def apply_mask(img: np.ndarray, img_hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray):
    mask = cv2.inRange(img_hsv, lower, upper)
    img_result = cv2.bitwise_and(img, img, mask=mask)
    return img_result


def get_filtered_by_colors_image(img, lower: np.ndarray, upper: np.ndarray):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if lower[0] >= 0:
        img_result = apply_mask(img, img_hsv, lower, upper)
    else:
        lower1 = lower.copy()
        upper1 = upper.copy()
        lower2 = lower1.copy()
        upper2 = upper1.copy()

        lower2[0] = 179 + lower[0]
        lower1[0] = 0
        upper2[0] = 179

        img_result1 = apply_mask(img, img_hsv, lower1, upper1)
        img_result2 = apply_mask(img, img_hsv, lower2, upper2)
        img_result = img_result1 + img_result2

    return img_result


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


def detection_tools_hsv(filename):
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
    cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

    img = cv2.imread(filename)
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

        cv2.imshow("Color detection", imgResult)
        cv2.waitKey(1)
        if cv2.waitKey(1) == 27:# & 0xFF == ord('q'):
            break
    print(f"lower = [{lower[0]}, {lower[1]}, {lower[2]}]")
    print(f"upper = [{upper[0]}, {upper[1]}, {upper[2]}]")


if __name__ == '__main__':
    detection_tools_hsv("test_images\\f_022611fa253a07ec.jpg")
# lower = [0, 0, 0]
# upper = [179, 86, 115]
