import numpy as np
import cv2
import re
import time
import requests
from bs4 import BeautifulSoup
import json
import sys

from color_detection import get_filtered_by_colors_image
from border_detection import get_all_approx_contours, get_bounding_boxes, preprocessing, PreprocessingParams
from text_detection import get_text_data, get_text

example_url = "https://www.tradingview.com/x/nShwrpHU/"

pre_params_for_text = PreprocessingParams(canny_thresholds=(30, 60),
                                          gauss_kernel_size=None,
                                          morph_kernel_size=(1, 1))


class EntityInfo:
    def __init__(self, lower: list,
                 upper: list,
                 params: PreprocessingParams or None,
                 searching_zone: tuple):
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.params = params
        self.searching_zone = searching_zone


red_price_lower = [0, 190, 0]
red_price_upper = [40, 255, 255]

green_price_lower = [85, 0, 125]
green_price_upper = [90, 255, 255]

gray_price_lower = [114, 15, 0]
gray_price_upper = [116, 40, 138]

red_area_lower = [143, 0, 0]
red_area_upper = [163, 255, 255]

green_area_lower = [79, 0, 0]
green_area_upper = [92, 106, 61]

price_searching_zone = ((0, 1), (0.9, 1))
area_searching_zone = ((0, 1), (0, 1))

red_price_info = EntityInfo(red_price_lower, red_price_upper, pre_params_for_text, price_searching_zone)
green_price_info = EntityInfo(green_price_lower, green_price_upper, pre_params_for_text, price_searching_zone)
gray_price_info = EntityInfo(gray_price_lower, gray_price_upper, pre_params_for_text, price_searching_zone)

red_area_info = EntityInfo(red_area_lower, red_area_upper, None, area_searching_zone)
green_area_info = EntityInfo(green_area_lower, green_area_upper, None, area_searching_zone)


def get_image_using_url(original_url: str) -> np.ndarray:
    """return the image from the standard page that the url points to"""

    response = requests.get(original_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    img_url = soup.find('img').get('src')

    img_response = requests.get(img_url)
    img_arr = np.asarray(bytearray(img_response.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)

    return img


# def preprocessing(img: np.ndarray) -> np.ndarray:
#     """use some filters to transform an image for better text recognition"""
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edged = cv2.Canny(gray, 30, 60)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
#     closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
#
#     cv2.imshow('img', cv2.resize(closed, (640, 640)))
#     cv2.waitKey(0)
#     return closed


def get_data_from_boxes(img: np.ndarray, lower: list, upper: list, patterns: tuple = (r"\d{1,}[.]\d{1,}",)) -> float:
    """
    filter image by colors and then find rectangles in this image,
    find their bounding boxes and recognize text in boxes which fulfill the required patterns
    """

    filtered_img = get_filtered_by_colors_image(img, lower, upper)
    approx_contours = get_all_approx_contours(filtered_img)
    bboxes = get_bounding_boxes(approx_contours)

    valid_text = []
    for bbox in bboxes:
        x, y, w, h = bbox
        if 1 <= w/h <= 5 and w*h > 16:
            cropped_img = img[y:y + h, x:x + w]
            preprocessed_img = preprocessing(cropped_img, pre_params_for_text)
            text = get_text_data(preprocessed_img)['text']
            valid_words = []
            for word in text:
                for pattern in patterns:
                    if re.search(pattern, word):
                        valid_words.append(word)
            if len(valid_words) > 0:
                valid_text.append(valid_words)
    for elem in valid_text:
        if len(elem) == 1:
            return elem[0]
    return 0.0


def get_current_area(img):
    """define in which area the current price is"""
    red_areas_img = get_filtered_by_colors_image(img, red_area_lower, red_area_upper)
    green_areas_img = get_filtered_by_colors_image(img, green_area_lower, green_area_upper)
    areas_img = red_areas_img + green_areas_img
    no_area_is_reached = True
    coord_x = img.shape[1] - 1
    coord_y = img.shape[0] // 2
    while no_area_is_reached:
        if img[coord_y, coord_x][0] != 0 and img[coord_y, coord_x][1] != 0 and img[coord_y, coord_x][2] != 0:
            if img[coord_y, coord_x][1] > img[coord_y, coord_x][2]:
                return "green"
            else:
                return "red"
        coord_x -= 1


def get_title(img):
    img_for_title = img[:80, :]
    text = get_text(img_for_title)
    title = text.split('\n')[1].split(',')[0].split(':')[1]
    return title


def write_into_json(filename, values, keys):
    data_dict = dict()
    for i in range(len(keys)):
        data_dict[keys[i]] = values[i]
    with open(filename, "w") as file:
        json.dump(data_dict, file, indent=4)


def main():
    """The main entry point of the application"""
    # print(sys.argv)
    start_time = time.time()
    # img = get_image_using_url(example_url)
    img = cv2.imread("example.png")
    print("Файл \'example.png\' открыт")
    img_for_prices = img[:, int(0.9 * img.shape[1]):img.shape[1]]

    print(f"Пара: {get_title(img)}")
    print(f"Текущая область сверху: {get_current_area(img)}")

    gray_data = get_data_from_boxes(img_for_prices, gray_price_lower, gray_price_upper)
    print(f"Серая цена: {gray_data}")

    red_data = get_data_from_boxes(img_for_prices, red_price_lower, red_price_upper)
    print(f"Красная цена: {red_data}")

    green_data = get_data_from_boxes(img_for_prices, green_price_lower, green_price_upper, (r"\d{1,}[.]\d{1,}",
                                                                                            r"\d{2}:\d{2}:\d{2}"))
    print(f"Зеленая цена: {green_data}")

    print("Время работы: {:.2f} с".format(time.time() - start_time))


if __name__ == '__main__':
    main()


