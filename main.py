import numpy as np
import cv2
from collections import namedtuple
import re
import time
import sys
import os

from config import *
from color_detection import get_filtered_by_colors_image, get_mean_color, get_nearest_mean_color
from border_detection import get_all_approx_contours, get_bounding_boxes, preprocessing_for_border_detection, find_contour_with_the_biggest_area
from border_detection import get_borders_of_vertical_scale
from text_recognition import get_digit_only_text_data, get_text, preprocessing_for_text_recognition, TextCash
from input_output import get_image_using_url, write_into_json

example_url = "https://www.tradingview.com/x/nShwrpHU/"

PriceInfo = namedtuple("PriceInfo", ["lower", "upper", "mean_color"])

red_price_info = PriceInfo(np.array(red_price_lower), np.array(red_price_upper), 'red_price_mean_color')
green_price_info = PriceInfo(np.array(green_price_lower), np.array(green_price_upper), 'green_price_mean_color')
gray_price_info = PriceInfo(np.array(gray_price_lower), np.array(gray_price_upper), 'gray_price_mean_color')
white_price_info = PriceInfo(np.array(white_price_lower), np.array(white_price_upper), 'white_price_mean_color')

PriceResult = namedtuple("PriceResult", ["value", "x", "y", "w", "h"])
AllPriceResult = namedtuple("AllPriceResult", ['red_data', 'green_data', 'gray_data', 'white_data'])


current_price_result = PriceResult('', 0, 0, 0, 0)


def get_text_data_from_boxes(img, bboxes, mean_color, text_cash: TextCash):
    global current_price_result
    price_pattern = r"\d{1,}[.]\d{1,}"
    time_pattern = r"\d{2}[:]\d{2}"
    price_result = []
    for bbox in bboxes:
        x, y, w, h = bbox

        cash_checking_result = text_cash.check(bbox)
        if cash_checking_result != -1:
            if text_cash.text[cash_checking_result] != '':
                # if ':' not in text_cash.text[cash_checking_result]:
                price_result.append(PriceResult(text_cash.text[cash_checking_result], x, y, w, h))
                continue

        if 1.8 <= w / h <= 2.1 and w * h > 64:
            pass
            # cropped_img = img[y:y + h, x:x + w]
            #
            # current_mean_color = get_mean_color(cropped_img)
            # if get_nearest_mean_color(current_mean_color) == 'blue_price_mean_color':
            #     break
            #
            # cropped_img1 = cropped_img[:cropped_img.shape[1] // 2, :]
            # cropped_img2 = cropped_img[cropped_img.shape[1] // 2:, :]
            #
            # preprocessed_img_list1 = preprocessing_for_text_recognition(cropped_img1)
            # preprocessed_img_list2 = preprocessing_for_text_recognition(cropped_img2)
            # search_price = False
            # search_time = False
            #
            # for preprocessed_img1 in preprocessed_img_list1:
            #     chars = get_digit_only_text_data(preprocessed_img1)['text']
            #     text = ''
            #     for ch in chars:
            #         text += ch
            #     search_price = re.search(price_pattern, text)
            # for preprocessed_img2 in preprocessed_img_list2:
            #     chars = get_digit_only_text_data(preprocessed_img2)['text']
            #     text = ''
            #     for ch in chars:
            #         text += ch
            #     search_time = re.search(time_pattern, text)
            # if search_price and search_time:
            #     current_price_result = PriceResult(f"{search_time.group(0)}", x, y, w, h) # разобраться со временем
            #     text_cash.add(bbox, f"{search_time.group(0)}")

        elif 2.5 <= w / h <= 5 and w * h > 32:
            valid_text = ""
            cropped_img = img[y:y + h, x:x + w]
            current_mean_color = get_mean_color(cropped_img)
            if get_nearest_mean_color(current_mean_color) != mean_color:
                break
            preprocessed_img_list = preprocessing_for_text_recognition(cropped_img)

            search = False
            for preprocessed_img in preprocessed_img_list:
                # preprocessed_img = preprocessing(cropped_img, params)
                # cv2.imshow('img', preprocessed_img)
                # cv2.waitKey(0)

                chars = get_digit_only_text_data(preprocessed_img)['text']
                text = ''
                for ch in chars:
                    text += ch

                search = re.search(price_pattern, text)
                if search and len(search.group(0)) > len(valid_text):
                    valid_text = search.group(0)
            if search:
                price_result.append(PriceResult(valid_text, x, y, w, h))

            if valid_text != '':
                text_cash.add(bbox, valid_text)
    return price_result, text_cash


def filter_result_text(price_result_list: list):# переработать
    # print(price_result_list)
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


def get_price_data(img: np.ndarray, price_info: PriceInfo):
    """
    filter image by colors and then find rectangles in this image,
    find their bounding boxes and recognize text in boxes which fulfill the required patterns
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


def get_ticker(img):
    ticker_pattern = r'[A-Z]{2,}[:-][A-Z]{2,}'
    img_for_ticker = img[:int(img.shape[0] * 0.25), :int(img.shape[1] * 0.5)]
    filtered_img = get_filtered_by_colors_image(img_for_ticker, np.array(ticker_lower), np.array(ticker_upper))
    text = get_text(filtered_img)

    search = re.search(ticker_pattern, text)
    if search:
        ticker = search.group(0)

        if ':' in ticker:
            ticker = ticker.split(':')[1]
        elif '-' in ticker:
            ticker = ticker.split('-')[1]
    else:
        ticker = 'None'
    # ticker = text.strip().split('\n')[1].split(',')[0].split(':')[1]
    return ticker


def prepare_image_for_price(img, width):
    img_result = img.copy()
    for x in range(width):
        for y in range(img.shape[0]):
            for canal in range(3):
                img_result[y, x][canal] = 0
    return img_result


def highlight_prices(img, border, all_price_result: AllPriceResult, ticker):

    img = cv2.putText(img, ticker, (int(img.shape[1] * 0.25), int(img.shape[0] * 0.25)),
                      cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
    img = cv2.line(img, (border, 0), (border, img.shape[0] - 1), (0, 255, 255))

    line_lenght = 900
    colors = [(0, 0, 255), (0, 255, 0), (128, 128, 128), (255, 255, 255)]
    for i in range(len(all_price_result)):
        price_result = all_price_result[i]
        color = colors[i]
        for price in price_result:
            x = price.x
            y = price.y
            w = price.w
            h = price.h
            value = price.value
            img = cv2.rectangle(img, (x + border - 7, y), (x + border - 7 + w, y + h), color, 2)
            img = cv2.line(img, (x + border - 7, y), (x + border - 7 - line_lenght, y), color)
            img = cv2.putText(img, value, (x + border - 7 - line_lenght, y - 5),
                              cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
    cv2.imshow('result', cv2.resize(img, (1280, 768)))
    cv2.waitKey(1)
    return img


def delete_intersecting_and_small(all_price_result: AllPriceResult):
    common_list = []
    mean = 0
    for price_result in all_price_result:
        for price in price_result:
            common_list.append(price.w * price.h)
    mean = np.array(common_list).mean()

    index_blacklists = []
    for price_result in all_price_result:
        index_blacklist = []
        for i in range(len(price_result)):
            if price_result[i].w * price_result[i].h < 0.5 * mean:
                index_blacklist.append(i)
        index_blacklists.append(index_blacklist)

    for i in range(len(all_price_result)):
        index_blacklist = index_blacklists[i]
        if len(index_blacklist) != 0:
            all_price_result[i].pop(*index_blacklist[::-1])

    return all_price_result


def test():
    img_paths = os.listdir("test_images")
    # print(img_paths)
    start_time = time.time()
    for path in img_paths:
        try:
            print("##################")
            print(path)
            img = cv2.imread(f"test_images\\{path}")
            ticker = get_ticker(img)
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
            all_price_result = delete_intersecting_and_small(all_price_result)
            img = highlight_prices(img, borders_for_prices[0], all_price_result, ticker)
            cv2.imwrite(f'results\\{path}', img)
        except:
            print(f"invalid file{path}")
    print("Время работы: {:.2f} с".format(time.time() - start_time))


def main():
    """The main entry point of the application"""

    # print(sys.argv)
    start_time = time.time()

    img = cv2.imread("test_images\\f_336611fa2539a0a1.jpg")
    # cv2.imshow('img', cv2.resize(img, (640, 640)))
    # cv2.waitKey(0)
    print("Файл открыт")
    borders_for_prices = get_borders_of_vertical_scale(img)
    img_for_prices = prepare_image_for_price(img[:, borders_for_prices[0]-7: img.shape[1]], 9)

    ticker = get_ticker(img)
    print(f"Тикер: {get_ticker(img)}")
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
    all_price_result = delete_intersecting_and_small(all_price_result)

    highlight_prices(img, borders_for_prices[0], all_price_result, ticker)
    print("Время работы: {:.2f} с".format(time.time() - start_time))


if __name__ == '__main__':
    main()

# ПЛАНЫ
# обработать пересечение
# найти текущую цену
#
