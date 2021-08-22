import numpy as np
import cv2
from collections import namedtuple
import re
import time
import sys
import os

from config import *
from color_detection import get_filtered_by_colors_image
from border_detection import get_all_approx_contours, get_bounding_boxes, preprocessing_for_border_detection, PreprocessingParams, find_contour_with_the_biggest_area
from border_detection import get_borders_of_vertical_scale
from text_detection import get_text_data, get_text, preprocessing_for_text_recognition
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


pre_params_for_text = PreprocessingParams(canny_thresholds=(30, 60),
                                          gauss_kernel_size=None,
                                              morph_kernel_size=(1, 1))

red_price_info = EntityInfo(red_price_lower, red_price_upper, pre_params_for_text, price_searching_zone)
green_price_info = EntityInfo(green_price_lower, green_price_upper, pre_params_for_text, price_searching_zone)
gray_price_info = EntityInfo(gray_price_lower, gray_price_upper, pre_params_for_text, price_searching_zone)
white_price_info = EntityInfo(white_price_lower, white_price_upper, pre_params_for_text, price_searching_zone)

pre_params_for_areas = PreprocessingParams(canny_thresholds=(30, 60),
                                               gauss_kernel_size=(3, 3),
                                               morph_kernel_size=(3, 3))

red_area_info = EntityInfo(red_area_lower, red_area_upper, pre_params_for_areas, area_searching_zone)
green_area_info = EntityInfo(green_area_lower, green_area_upper, pre_params_for_areas, area_searching_zone)


PriceResult = namedtuple("PriceResult", ["value", "x", "y", "w", "h"])
AllPriceResult = namedtuple("AllPriceResult", ['red_data', 'green_data', 'gray_data', 'white_data'])


def get_text_data_from_boxes(img,
                             bboxes):

    price_pattern = r"\d{1,}[.]\d{1,}"
    valid_text = ""
    price_result = []
    for bbox in bboxes:
        x, y, w, h = bbox
        if 1 <= w / h <= 5 and w * h > 16:
            cropped_img = img[y:y + h, x:x + w]
            preprocessed_img = preprocessing_for_text_recognition(cropped_img)
            # preprocessed_img = preprocessing(cropped_img, params)

            # cv2.imshow('img', preprocessed_img)
            # cv2.waitKey(0)

            chars = get_text_data(preprocessed_img)['text']
            text = ''
            for ch in chars:
                text += ch
            # print(text)

            search = re.search(price_pattern, text)
            if search:
                valid_text = search.group(0)
                price_result.append(PriceResult(valid_text, x, y, w, h))

    return price_result


def filter_result_text(price_result_list: list):# переработать
    if len(price_result_list) == 0:
        return None
    len_list = [len(price_result) for price_result in price_result_list]

    max_len_price_result_list = []
    max_len = max(len_list)
    for price_result in price_result_list:
        if len(price_result) == max_len:
            max_len_price_result_list.append(price_result)
    assert len(max_len_price_result_list) > 0

    if max_len == 1:
        for price_result in max_len_price_result_list:
            if price_result[0] != '':
                return price_result
        return max_len_price_result_list[0]

    # print(max_len_price_result_list)
    for i in range(len(max_len_price_result_list) - 1):
        price_result_list = sorted(max_len_price_result_list[i], key=lambda x: float(x.value))
        for j in range(max_len):
            if max_len_price_result_list[i][j].y < max_len_price_result_list[i][j].y:
                break
        return max_len_price_result_list[0]

    return max_len_price_result_list[0]


def get_price_data(img: np.ndarray, price_info: EntityInfo):
    """
    filter image by colors and then find rectangles in this image,
    find their bounding boxes and recognize text in boxes which fulfill the required patterns
    """

    lower = price_info.lower
    upper = price_info.upper
    params = price_info.params

    filtered_img = get_filtered_by_colors_image(img, lower, upper)
    approx_contours_list = get_all_approx_contours(filtered_img)
    bboxes_list = get_bounding_boxes(approx_contours_list)

    price_resilt_list = []
    for bboxes in bboxes_list:
        price_result = get_text_data_from_boxes(img, bboxes)
        price_resilt_list.append(price_result)
    final = filter_result_text(price_resilt_list)
    return final


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


def prepare_image_for_price(img, width):
    for x in range(width):
        for y in range(img.shape[0]):
            for canal in range(3):
                img[y, x][canal] = 0
    return img


def highlight_prices(img, all_price_result: AllPriceResult):
    for price_result in all_price_result:
        for price in price_result:
            x = price.x
            y = price.y
            w = price.w
            h = price.h
            value = price.value
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 4)
            cv2.putText(img, value, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imshow('result', cv2.resize(img, (140, 800)))
    cv2.waitKey(200)


def test():
    img_paths = os.listdir("test_images")
    print(img_paths)
    for path in img_paths:
        try:
            img = cv2.imread(f"test_images\\{path}")
            borders_for_prices = get_borders_of_vertical_scale(img)
            img_for_prices = prepare_image_for_price(img[:, borders_for_prices[0] - 7: img.shape[1]], 7)

            red_data = get_price_data(img_for_prices, red_price_info)
            print(f"Красная цена: {red_data}")

            gray_data = get_price_data(img_for_prices, gray_price_info)
            print(f"Серая цена: {gray_data}")

            green_data = get_price_data(img_for_prices, green_price_info)
            print(f"Зеленая цена: {green_data}")

            white_data = get_price_data(img_for_prices, white_price_info)
            print(f"Белая цена: {white_data}")

            all_price_result = AllPriceResult(red_data, green_data, gray_data, white_data)
            highlight_prices(img_for_prices, all_price_result)
        except:
            print(f"invalid file {path}")


def main():
    """The main entry point of the application"""

    # print(sys.argv)
    start_time = time.time()

    img = cv2.imread("test_images\\f_044611fa24868e08.png")
    # cv2.imshow('img', cv2.resize(img, (640, 640)))
    # cv2.waitKey(0)
    print("Файл \'example.png\' открыт")
    borders_for_prices = get_borders_of_vertical_scale(img)
    # borders_for_prices.append(img.shape[1])         # временно
    # borders_for_prices = borders_for_prices[::-1]   # временно

    img_for_prices = prepare_image_for_price(img[:, borders_for_prices[1]-7: borders_for_prices[0]], 7)

    # print(f"Тикер: {get_title(img)}")
    # print(f"Текущая область сверху: {get_current_area(img)}")

    red_data = get_price_data(img_for_prices, red_price_info)
    print(f"Красная цена: {red_data}")

    gray_data = get_price_data(img_for_prices, gray_price_info)
    print(f"Серая цена: {gray_data}")

    green_data = get_price_data(img_for_prices, green_price_info)
    print(f"Зеленая цена: {green_data}")

    white_data = get_price_data(img_for_prices, white_price_info)
    print(f"Белая цена: {white_data}")

    all_price_result = AllPriceResult(red_data, green_data, gray_data, white_data)
    highlight_prices(img_for_prices, all_price_result)
    print("Время работы: {:.2f} с".format(time.time() - start_time))


if __name__ == '__main__':
    test()


