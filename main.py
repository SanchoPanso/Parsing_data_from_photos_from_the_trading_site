import numpy as np
import cv2
import time
import sys
import os

from config import *
from border_detection import get_borders_of_vertical_scale
from input_output import get_image_using_url, write_into_json

from searching_for_definite_objects import AllPriceResults, PriceResult
from searching_for_definite_objects import red_price_info, green_price_info, gray_price_info, white_price_info
from searching_for_definite_objects import prepare_image_for_price
from searching_for_definite_objects import get_price_data, define_direction, delete_intersecting_and_small
from searching_for_definite_objects import get_ticker

from debug_and_testing import highlight_prices


def main():
    """The main entry point of the application"""

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
    main()

