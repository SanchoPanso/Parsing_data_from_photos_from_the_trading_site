import cv2
import numpy as np
from config import mean_colors
from input_output import get_image_using_url


def get_color_filtered_image(img, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
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


def get_mean_color(img: np.ndarray) -> list:
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


def check_color_proximity(mean_color_key: str, current_color: list) -> bool:
    # create dictionary of distances from current_color to all mean colors in config
    # distance - an euclidean distance in 3-dim space of colors
    distances = dict()
    for key in mean_colors.keys():
        mean_color = mean_colors[key]
        distance = sum([(current_color[canal] - mean_color[canal])**2 for canal in range(3)])
        distances[key] = distance

    # if minimal distance correspond to supposed mean color
    # return True
    min_distance = min(distances.values())
    if min_distance == distances[mean_color_key]:
        return True

    # if distance to supposed mean color is not minimal, but does not extremely differs from real minimal
    # then we can consider, that supposed mean color is correspond (however, maybe it is not needed)
    if min_distance / distances[mean_color_key] > 0.8:
        return True

    return False


# def check_color_proximity(mean_color_key: str, current_color: list) -> bool:
#     mean_colors_keys_and_values = []
#     for key in mean_colors.keys():
#         mean_colors_keys_and_values.append([key, mean_colors[key]])
#
#     keys_and_distances = []
#     for color in mean_colors_keys_and_values:
#         distance = 0
#         for canal in range(3):
#             distance += (current_color[canal] - color[1][canal]) ** 2
#         keys_and_distances.append([color[0], distance])
#
#     keys_and_distances = sorted(keys_and_distances, key=lambda x: x[1])
#     if keys_and_distances[0][0] == mean_color_key:
#         return True
#
#     for i in range(1, len(keys_and_distances)):
#         if keys_and_distances[i][0] == mean_color_key:
#             # if distances[0][1] / distances[i][1] > 0.8: # возможно, поменять
#                 return True
#     return False


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


def detection_tools_hsv(filename: str):
    """
    Auxiliary function that helps to define color mask.
    IT IS NOT USED IN THE PROGRAM, but can help you to correct configs.
    :param filename: path to image
    """

    # create windows
    cv2.namedWindow("TrackBars")
    cv2.resizeWindow("TrackBars", 640, 240)
    cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, lambda x: None)
    cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, lambda x: None)
    cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, lambda x: None)
    cv2.createTrackbar("Sat Max", "TrackBars", 255, 255, lambda x: None)
    cv2.createTrackbar("Val Min", "TrackBars", 0, 255, lambda x: None)
    cv2.createTrackbar("Val Max", "TrackBars", 255, 255, lambda x: None)

    # reading and preparing of an image
    # img = get_image_using_url("https://www.tradingview.com/x/JopwW6IR/")
    img = cv2.imread(filename)
    img = img[:, int(img.shape[1]*0.5):]
    img = cv2.resize(img, (640, 640))

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # In loop get hsv limits (lower, upper) from trackbar and apply mask to image
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
        # close windows on press key "escape"
        if cv2.waitKey(1) == 27:     # & 0xFF == ord('q'):
            break

    # after closing windows print the last hsv limits of chosen mask
    print(f"lower = [{lower[0]}, {lower[1]}, {lower[2]}]")
    print(f"upper = [{upper[0]}, {upper[1]}, {upper[2]}]")


if __name__ == '__main__':
    detection_tools_hsv("example.jpg")

