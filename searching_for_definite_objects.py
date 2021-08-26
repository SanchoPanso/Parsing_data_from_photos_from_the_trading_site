"""
This module contains function for searching for definite object (rectangles with price, text with ticker) and
for filtration results of searching. For this purposes the module uses functions from text_recognition.py,
border_detection.py, color_detection.py.
"""
import numpy as np
import cv2
from collections import namedtuple
import re

from config import *
from preprocessing import preprocessing_for_text_recognition
from color_detection import get_filtered_by_colors_image, get_mean_color, check_color_proximity
from border_detection import get_all_approx_contours, get_bounding_boxes
from text_recognition import get_digit_only_text_data, get_text, TextCash

example_url = "https://www.tradingview.com/x/nShwrpHU/"

PriceInfo = namedtuple("PriceInfo", ["lower", "upper", "mean_color"])

red_price_info = PriceInfo(np.array(red_price_lower), np.array(red_price_upper), 'red_price_mean_color')
green_price_info = PriceInfo(np.array(green_price_lower), np.array(green_price_upper), 'green_price_mean_color')
gray_price_info = PriceInfo(np.array(gray_price_lower), np.array(gray_price_upper), 'gray_price_mean_color')
white_price_info = PriceInfo(np.array(white_price_lower), np.array(white_price_upper), 'white_price_mean_color')


class PriceResult:
    def __init__(self, value, x, y, w, h, reliability):
        self.value = value
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.reliability = reliability

    def __repr__(self):
        return f"PriceResults(value={self.value}, x={self.x}, y={self.y}, " \
               f"w={self.w}, h={self.h}, reliability={self.reliability})"

    def __str__(self):
        return f"PriceResults(value={self.value}, x={self.x}, y={self.y}, " \
               f"w={self.w}, h={self.h}, reliability={self.reliability})"


def get_text_data_from_boxes(img: np.array, bboxes: list, mean_color_key: str, text_cash: TextCash,
                             extra_cropping_width: int = 0):
    price_pattern = r"\d{1,}[.]\d{1,}"
    # time_pattern = r"\d{2}[:]\d{2}"
    price_result = []
    for bbox in bboxes:
        x, y, w, h = bbox

        cash_checking_result = text_cash.check(bbox)
        if cash_checking_result >= 0:
            price_result.append(PriceResult(text_cash.text[cash_checking_result], x, y, w, h, True))
            continue
        elif cash_checking_result == -2:
            continue

        if 2.15 <= w / h <= 5.5 and w * h > 32:
            valid_text = ""
            cropped_img = img[y:y + h, x + extra_cropping_width:x + w]
            current_mean_color = get_mean_color(cropped_img)
            if not check_color_proximity(mean_color_key, current_mean_color):
                break

            preprocessed_img_list = preprocessing_for_text_recognition.preprocess(cropped_img)

            search = False
            for preprocessed_img in preprocessed_img_list:
                if debug_mode:
                    cv2.imshow('img', preprocessed_img)
                    cv2.waitKey(dbg_delay)

                chars = get_digit_only_text_data(preprocessed_img)['text']
                text = ''
                for ch in chars:
                    text += ch

                search = re.search(price_pattern, text)
                if search and len(search.group(0)) > len(valid_text):
                    valid_text = search.group(0)
            if search:
                price_result.append(PriceResult(valid_text, x, y, w, h, True))

            # if valid_text != '':
            text_cash.upsert(bbox, valid_text)
    return price_result, text_cash


def filter_result_text(price_result_list: list):
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

    for i in range(len(max_len_price_result_list) - 1):
        price_result_list = sorted(max_len_price_result_list[i], key=lambda x: float(x.value))
        for j in range(max_len):
            if max_len_price_result_list[i][j].y < max_len_price_result_list[i][j].y:
                break
        return max_len_price_result_list[0]

    return max_len_price_result_list[0]


def get_price_data(img: np.ndarray, price_info: PriceInfo):
    """
    filter image by colors and then find rectangles in this image,
    find their bounding boxes and recognize text in boxes which fulfill the required price pattern
    """

    lower = price_info.lower
    upper = price_info.upper
    mean_color = price_info.mean_color

    filtered_img = get_filtered_by_colors_image(img, lower, upper)
    approx_contours_list = get_all_approx_contours(filtered_img)
    bboxes_list = get_bounding_boxes(approx_contours_list)

    price_result_list = []
    text_cash = TextCash()
    for bboxes in bboxes_list:
        price_result, text_cash = get_text_data_from_boxes(img, bboxes, mean_color, text_cash)
        price_result_list.append(price_result)
    final = filter_result_text(price_result_list)
    return final


def get_ticker(img: np.ndarray):
    """
    get ticker with cropping original image and searching the ticker pattern
    """
    ticker_pattern = r'[A-Z]{2,}[:-][A-Z]{2,}'
    img_for_ticker = img[:int(img.shape[0] * 0.25), :int(img.shape[1] * 0.5)]
    filtered_img = get_filtered_by_colors_image(img_for_ticker, np.array(ticker_lower), np.array(ticker_upper))
    thr_img = cv2.threshold(cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY), 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    text = get_text(thr_img)

    search = re.search(ticker_pattern, text)
    if search:
        ticker = search.group(0)

        if ':' in ticker:
            ticker = ticker.split(':')[1]
        elif '-' in ticker:
            ticker = ticker.split('-')[1]
    else:
        ticker = 'None'
    return ticker


def prepare_image_for_price(img: np.ndarray, border: int, width: int):
    """fill with black color the left border of the cropped vertical scale"""
    img_result = img[:, border - width + 2: img.shape[1]].copy()
    for x in range(width):
        for y in range(img.shape[0]):
            for canal in range(3):
                img_result[y, x][canal] = 0
    return img_result


def delete_intersecting_and_small(all_price_results: dict):
    sqr_list = []
    coord_list = []
    for i in all_price_results.keys():
        for j in range(len(all_price_results[i]) - 1, -1, -1):
            sqr_list.append(all_price_results[i][j].w * all_price_results[i][j].h)

            x1 = all_price_results[i][j].x
            y1 = all_price_results[i][j].y
            w1 = all_price_results[i][j].w
            h1 = all_price_results[i][j].h
            for k in all_price_results.keys():
                stop_flag = False
                if stop_flag:
                    break
                if k == i:
                    continue
                for l in range(len(all_price_results[k])):

                    x2 = all_price_results[k][l].x
                    y2 = all_price_results[k][l].y
                    w2 = all_price_results[k][l].w
                    h2 = all_price_results[k][l].h

                    if x2 < x1 + w1 / 2 < x2 + w2 and y2 < y1 + h1 / 2 < y2 + h2 and w1 * h1 < w2 * h2:
                        all_price_results[i].pop(j)
                        stop_flag = True

    mean = np.array(sqr_list).mean()
    index_blacklists = dict()
    for i in all_price_results.keys():
        price_result = all_price_results[i]
        index_blacklist = []
        for j in range(len(price_result)):
            if price_result[j].w * price_result[j].h < 0.6 * mean:
                index_blacklist.append(j)
        index_blacklists[i] = index_blacklist

    for i in all_price_results.keys():
        index_blacklist = index_blacklists[i]
        if len(index_blacklist) != 0:
            for j in range(len(index_blacklist) - 1, -1, -1):
                all_price_results[i].pop(index_blacklist[j])

    return all_price_results


def define_direction(all_price_result: dict):

    red_data = all_price_result[red_price_key]
    green_data = all_price_result[green_price_key]
    gray_data = all_price_result[gray_price_key]

    if len(red_data) != 0 and len(green_data) != 0:
        if red_data[0].y > green_data[-1].y:
            return direction_up_sign
        else:
            return direction_down_sign

    if len(red_data) != 0 and len(gray_data) != 0:
        if red_data[0].y > gray_data[0].y:
            return direction_up_sign
        else:
            return direction_down_sign

    if len(gray_data) != 0 and len(green_data) != 0:
        if gray_data[0].y > green_data[-1].y:
            return direction_up_sign
        else:
            return direction_down_sign

    return direction_down_sign


def mark_wrong_price_results(all_price_results: dict):
    number_of_digits_before_dot_list = []
    number_of_digits_after_dot_list = []
    for key in all_price_results.keys():
        price_results = all_price_results[key]
        for price_result in price_results:
            digits_before_dot = price_result.value.split('.')[0]
            number_of_digits_before_dot_list.append(len(digits_before_dot))

            digits_after_dot = price_result.value.split('.')[1]
            number_of_digits_after_dot_list.append(len(digits_after_dot))

    digits_before_dot_stat = [0]*max(number_of_digits_before_dot_list)
    digits_after_dot_stat = [0]*max(number_of_digits_after_dot_list)

    for i in range(len(number_of_digits_after_dot_list)):
        digits_before_dot_stat[number_of_digits_before_dot_list[i] - 1] += 1
        digits_after_dot_stat[number_of_digits_after_dot_list[i] - 1] += 1

    digits_before_dot_mode = np.array(digits_before_dot_stat).argmax() + 1
    digits_after_dot_mode = np.array(digits_after_dot_stat).argmax() + 1

    for key in all_price_results.keys():
        for i in range(len(all_price_results[key])):

            price_result = all_price_results[key][i]
            digits_before_dot = price_result.value.split('.')[0]
            digits_after_dot = price_result.value.split('.')[1]

            if len(digits_before_dot) != digits_before_dot_mode:
                all_price_results[key][i].reliability = False
            if len(digits_after_dot) != digits_after_dot_mode:
                all_price_results[key][i].reliability = False

    return all_price_results
