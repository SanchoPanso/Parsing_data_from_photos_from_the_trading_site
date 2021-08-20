import numpy as np
import cv2
import re
import time
import sys

from config import *
from color_detection import get_filtered_by_colors_image
from border_detection import get_all_approx_contours, get_bounding_boxes, preprocessing, PreprocessingParams, find_contour_with_the_biggest_area
from text_detection import get_text_data, get_text
from input_output import get_image_using_url, write_into_json

example_url = "https://www.tradingview.com/x/nShwrpHU/"


class EntityInfo:
    def __init__(self, lower: list,
                 upper: list,
                 params: PreprocessingParams or None,
                 searching_zone: tuple):
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.params = params
        self.searching_zone = searching_zone


def get_data_from_boxes(img: np.ndarray, price_info: EntityInfo, patterns: tuple) -> float:
    """
    filter image by colors and then find rectangles in this image,
    find their bounding boxes and recognize text in boxes which fulfill the required patterns
    """

    lower = price_info.lower
    upper = price_info.upper
    params = price_info.params

    filtered_img = get_filtered_by_colors_image(img, lower, upper)
    approx_contours = get_all_approx_contours(filtered_img)
    bboxes = get_bounding_boxes(approx_contours)

    valid_text = []
    for bbox in bboxes:
        x, y, w, h = bbox
        if 1 <= w/h <= 5 and w*h > 16:
            cropped_img = img[y:y + h, x:x + w]
            preprocessed_img = preprocessing(cropped_img, params)
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


def get_current_area(img, red_area_info, green_area_info):
    """define in which area the current price is"""

    red_filtered_img = get_filtered_by_colors_image(img, red_area_info.lower, red_area_info.upper)
    red_contours = get_all_approx_contours(red_filtered_img, red_area_info.params)
    max_red_contour = find_contour_with_the_biggest_area(red_contours)

    green_filtered_img = get_filtered_by_colors_image(img, green_area_info.lower, green_area_info.upper)
    green_contours = get_all_approx_contours(green_filtered_img, green_area_info.params)
    max_green_contour = find_contour_with_the_biggest_area(green_contours)

    red_bbox, green_bbox = get_bounding_boxes([max_red_contour, max_green_contour])
    print(red_bbox)
    print(green_bbox)

    if red_bbox[1] < green_bbox[1]:
        return "red"
    else:
        return "green"


def get_title(img):
    img_for_title = img[:80, :]
    text = get_text(img_for_title)
    title = text.split('\n')[1].split(',')[0].split(':')[1]
    return title


def main():
    """The main entry point of the application"""

    pre_params_for_text = PreprocessingParams(canny_thresholds=(30, 60),
                                              gauss_kernel_size=None,
                                              morph_kernel_size=(1, 1))

    red_price_info = EntityInfo(red_price_lower, red_price_upper, pre_params_for_text, price_searching_zone)
    green_price_info = EntityInfo(green_price_lower, green_price_upper, pre_params_for_text, price_searching_zone)
    gray_price_info = EntityInfo(gray_price_lower, gray_price_upper, pre_params_for_text, price_searching_zone)

    pre_params_for_areas = PreprocessingParams(canny_thresholds=(30, 60),
                                               gauss_kernel_size=(3, 3),
                                               morph_kernel_size=(3, 3))

    red_area_info = EntityInfo(red_area_lower, red_area_upper, pre_params_for_areas, area_searching_zone)
    green_area_info = EntityInfo(green_area_lower, green_area_upper, pre_params_for_areas, area_searching_zone)

    # print(sys.argv)
    start_time = time.time()
    # img = get_image_using_url(example_url)
    img = cv2.imread("example.png")
    print("Файл \'example.png\' открыт")
    print(get_current_area(img, red_area_info, green_area_info))
    img_for_prices = img[:, int(0.9 * img.shape[1]):img.shape[1]]

    print(f"Пара: {get_title(img)}")
    # print(f"Текущая область сверху: {get_current_area(img)}")

    gray_data = get_data_from_boxes(img_for_prices, gray_price_info, (r"\d{1,}[.]\d{1,}",))
    print(f"Серая цена: {gray_data}")

    red_data = get_data_from_boxes(img_for_prices, red_price_info, (r"\d{1,}[.]\d{1,}", r"\d{2}:\d{2}:\d{2}"))
    print(f"Красная цена: {red_data}")

    green_data = get_data_from_boxes(img_for_prices, green_price_info, (r"\d{1,}[.]\d{1,}", r"\d{2}:\d{2}:\d{2}"))
    print(f"Зеленая цена: {green_data}")

    print("Время работы: {:.2f} с".format(time.time() - start_time))


if __name__ == '__main__':
    main()


