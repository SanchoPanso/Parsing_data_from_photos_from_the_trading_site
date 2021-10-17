import cv2
import numpy as np
from config import mean_colors
from input_output import get_image_using_url

def define_mean_color_tool():
    samples = ["red_sample", "green_sample", "gray_sample", "white_sample", "blue_sample"]
    for name in samples:
        sample = cv2.imread(f"color_samples\\{name}.png")
        mean = [0, 0, 0]
        for x in range(sample.shape[1]):
            for y in range(sample.shape[0]):
                for canal in range(3):
                    mean[canal] += sample[y, x][canal]
        for canal in range(3):
            mean[canal] /= sample.shape[0] * sample.shape[1]
        for canal in range(3):
            mean[canal] = int(mean[canal])
        image = np.zeros((500, 500, 3), dtype=np.uint8)
        for x in range(500):
            for y in range(500):
                for canal in range(3):
                    image[y, x][canal] = mean[canal]

        cv2.imshow(name, image)
        print(f"{name} = {mean}")
    cv2.waitKey(0)


def get_mean_color(img):
    width = img.shape[1]
    height = img.shape[0]
    mean = [0, 0, 0]
    for x in range(width):
        for y in range(height):
            for canal in range(3):
                mean[canal] += img[y, x][canal]
    for canal in range(3):
        mean[canal] = int(mean[canal] / (width * height))
    return mean


def check_color_proximity(mean_color_key: str, query_color_value: list):
    mean_colors_list = []
    for key in mean_colors.keys():
        mean_colors_list.append([key, mean_colors[key]])
    distances = []
    for c in mean_colors_list:
        distance = 0
        for canal in range(3):
            distance += (query_color_value[canal] - c[1][canal]) ** 2
        distances.append([c[0], distance])
    distances = sorted(distances, key=lambda x: x[1])
    if distances[0][0] == mean_color_key:
        return True
    for i in range(1, len(distances)):
        if distances[i][0] == mean_color_key:
            if distances[0][1] / distances[i][1] > 0.8: # возможно, поменять
                return True
    return False


def get_nearest_mean_color(current_mean_color):
    keys = mean_colors.keys()
    distances = dict()
    for key in keys:
        distance = 0
        for canal in range(3):
            distance += (current_mean_color[canal] - mean_colors[key][canal]) ** 2
        distances[key] = distance

    min_key = ''
    min_value = 0
    for key in keys:
        if min_value == 0 or min_value > distances[key]:
            min_key = key
            min_value = distances[key]

    return min_key


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


def empty(a):
    pass


def detection_tools_rgb(filename):
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    cv2.createTrackbar("R Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("R Max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("G Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("G Max", "TrackBars", 255, 255, empty)
    cv2.createTrackbar("B Min", "TrackBars", 0, 255, empty)
    cv2.createTrackbar("B Max", "TrackBars", 255, 255, empty)

    img = cv2.imread(filename)
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
        if cv2.waitKey(1) == 27:   # & 0xFF == ord('q'):
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

    img = get_image_using_url("https://www.tradingview.com/x/JopwW6IR/")
    img = img[:, int(img.shape[1]*0.5):]

    img = cv2.resize(img, (640, 640))
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
        if cv2.waitKey(1) == 27:     # & 0xFF == ord('q'):
            break
    print(f"lower = [{lower[0]}, {lower[1]}, {lower[2]}]")
    print(f"upper = [{upper[0]}, {upper[1]}, {upper[2]}]")


if __name__ == '__main__':
    detection_tools_hsv("example.jpg")

