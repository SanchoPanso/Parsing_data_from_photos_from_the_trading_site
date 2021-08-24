import numpy as np
import cv2
import time
import sys
import os

from config import *
from border_detection import get_all_approx_contours, get_bounding_boxes, get_borders_of_vertical_scale
from input_output import get_image_using_url, write_into_json

from searching_for_definite_objects import AllPriceResults, PriceResult
from searching_for_definite_objects import red_price_info, green_price_info, gray_price_info, white_price_info
from searching_for_definite_objects import prepare_image_for_price
from searching_for_definite_objects import get_price_data, define_direction, delete_intersecting_and_small
from searching_for_definite_objects import get_ticker


def highlight_prices(img, border, all_price_result: AllPriceResults, ticker, direction, delay):

    img = cv2.putText(img, 'Tiker: ' + ticker,
                      (int(img.shape[1] * 0.25), int(img.shape[0] * 0.25)),
                      cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    img = cv2.putText(img, 'Direction: ' + direction,
                      (int(img.shape[1] * 0.25), int(img.shape[0] * 0.3)),
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
    cv2.waitKey(delay)
    return img


def test_directory():
    img_paths = os.listdir("test_images")
    start_time = time.time()
    for path in img_paths:
        try:
            print("##################")
            print(path)
            img = cv2.imread(f"test_images\\{path}")
            ticker = get_ticker(img)
            borders_for_prices = get_borders_of_vertical_scale(img)
            img_for_prices = prepare_image_for_price(img[:, borders_for_prices[0] - 7: img.shape[1]], 9)

            red_data = get_price_data(img_for_prices, red_price_info)
            print(f"Красная цена: {red_data}")

            gray_data = get_price_data(img_for_prices, gray_price_info)
            print(f"Серая цена: {gray_data}")

            green_data = get_price_data(img_for_prices, green_price_info)
            print(f"Зеленая цена: {green_data}")

            white_data = get_price_data(img_for_prices, white_price_info)
            print(f"Белая цена: {white_data}")

            all_price_results = AllPriceResults(red_data, green_data, gray_data, white_data)
            all_price_results = delete_intersecting_and_small(all_price_results)
            direction = define_direction(all_price_results)
            img = highlight_prices(img, borders_for_prices[0], all_price_results, ticker, direction, 1)
            cv2.imwrite(f'results\\{path}', img)
        except Exception as e:
            print(e)
            print(f"There is a trouble with file{path}")
    print("Время работы: {:.2f} с".format(time.time() - start_time))
    cv2.waitKey(0)


def test_one_file():
    # print(sys.argv)
    start_time = time.time()

    img = cv2.imread("test_images\\f_754611fa24864966.png")

    print("Файл открыт")
    borders_for_prices = get_borders_of_vertical_scale(img)
    img_for_prices = prepare_image_for_price(img[:, borders_for_prices[0]-7: img.shape[1]], 9)

    ticker = get_ticker(img)
    print(f"Тикер: {get_ticker(img)}")

    red_data = get_price_data(img_for_prices, red_price_info)
    print(f"Красная цена: {red_data}")

    gray_data = get_price_data(img_for_prices, gray_price_info)
    print(f"Серая цена: {gray_data}")

    green_data = get_price_data(img_for_prices, green_price_info)
    print(f"Зеленая цена: {green_data}")

    white_data = get_price_data(img_for_prices, white_price_info)
    print(f"Белая цена: {white_data}")

    all_price_results = AllPriceResults(red_data, green_data, gray_data, white_data)
    all_price_results = delete_intersecting_and_small(all_price_results)
    direction = define_direction(all_price_results)

    print("Время работы: {:.2f} с".format(time.time() - start_time))
    highlight_prices(img, borders_for_prices[0], all_price_results, ticker, direction, 0)


if __name__ == '__main__':
    test_directory()
