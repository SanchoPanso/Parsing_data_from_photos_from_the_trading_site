import numpy as np
import cv2
import re
import time
import sys

from config import *
from color_detection import get_filtered_by_colors_image, increase_contrast
from border_detection import get_all_approx_contours, get_bounding_boxes, preprocessing, PreprocessingParams, find_contour_with_the_biggest_area
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

pre_params_for_areas = PreprocessingParams(canny_thresholds=(30, 60),
                                               gauss_kernel_size=(3, 3),
                                               morph_kernel_size=(3, 3))

red_area_info = EntityInfo(red_area_lower, red_area_upper, pre_params_for_areas, area_searching_zone)
green_area_info = EntityInfo(green_area_lower, green_area_upper, pre_params_for_areas, area_searching_zone)


def delete_artifacts(img):
    mean = [0, 0, 0]
    width = img.shape[1]
    height = img.shape[0]
    depth = 0
    black_borders_is_not_deleted = True
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    threshold = 140

    for x in range(width):
        for y in range(height):
            for canal in range(3):
                mean[canal] += img[y, x][canal]
    for canal in range(3):
        mean[canal] /= width * height

    while black_borders_is_not_deleted:
        black_borders_is_not_deleted = False
        for x in range(depth, width - depth):
            y = 0
            if img_hsv[y, x][2] < threshold:
                black_borders_is_not_deleted = True
                for canal in range(3):
                    img[y, x][canal] = mean[canal]

        for x in range(depth, width - depth):
            y = height - 1
            if img_hsv[y, x][2] < threshold:
                black_borders_is_not_deleted = True
                for canal in range(3):
                    img[y, x][canal] = mean[canal]

        for y in range(depth, height - depth):
            x = 0
            if img_hsv[y, x][2] < threshold:
                black_borders_is_not_deleted = True
                for canal in range(3):
                    img[y, x][canal] = mean[canal]

        for y in range(depth, height - depth):
            x = width - 1
            if img_hsv[y, x][2] < threshold:
                black_borders_is_not_deleted = True
                for canal in range(3):
                    img[y, x][canal] = mean[canal]
        depth += 1
    return img


def get_data_from_boxes(img: np.ndarray, price_info: EntityInfo, patterns: tuple):
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
            preprocessed_img = preprocessing_for_text_recognition(cropped_img)
            # preprocessed_img = preprocessing(cropped_img, params)

            cv2.imshow('img', preprocessed_img)
            cv2.waitKey(0)

            # preprocessed_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            # preprocessed_img = cv2.resize(preprocessed_img, (preprocessed_img.shape[1] * 4,
            #                                                  preprocessed_img.shape[0] * 4))
            #
            #
            # preprocessed_img = cv2.threshold(preprocessed_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            chars = get_text_data(preprocessed_img)['text']

            text = ''
            for ch in chars:
                text += ch
            print(text)

            valid_words = []
            for pattern in patterns:
                search = re.search(pattern, text)
                if search:
                    valid_words.append(search.group(0))
            if len(valid_words) > 0:
                valid_text.append(valid_words)
    return valid_text


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


def main():
    """The main entry point of the application"""

    # print(sys.argv)
    start_time = time.time()
    # img = get_image_using_url(example_url)
    img = cv2.imread("example2.jpg")
    # cv2.imshow('img', cv2.resize(img, (640, 640)))
    # cv2.waitKey(0)
    print("Файл \'example.png\' открыт")
    borders_for_prices = get_borders_of_vertical_scale(img)
    # borders_for_prices.append(img.shape[1])         # временно
    # borders_for_prices = borders_for_prices[::-1]   # временно
    img_for_prices = prepare_image_for_price(img[:, borders_for_prices[1]-7: borders_for_prices[0]], 7)

    print(f"Тикер: {get_title(img)}")
    # print(f"Текущая область сверху: {get_current_area(img)}")

    red_data = get_data_from_boxes(img_for_prices, red_price_info, (r"\d{1,}[.]\d{1,}", r"\d{2}:\d{2}:\d{2}"))
    print(f"Красная цена: {red_data}")

    gray_data = get_data_from_boxes(img_for_prices, gray_price_info, (r"\d{1,}[.]\d{1,}",))
    print(f"Серая цена: {gray_data}")

    green_data = get_data_from_boxes(img_for_prices, green_price_info, (r"\d{1,}[.]\d{1,}", r"\d{2}:\d{2}:\d{2}"))
    print(f"Зеленая цена: {green_data}")

    print("Время работы: {:.2f} с".format(time.time() - start_time))


if __name__ == '__main__':
    main()


