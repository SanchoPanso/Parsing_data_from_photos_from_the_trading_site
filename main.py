import os
from config import *
from border_detection import get_borders_of_vertical_scale

from input_output import write_into_json, prepare_dict_for_output
from input_output import get_image, get_image_using_path

from searching_for_definite_objects import PriceResult
from searching_for_definite_objects import red_price_info, green_price_info, gray_price_info, white_price_info
from searching_for_definite_objects import prepare_image_for_price
from searching_for_definite_objects import get_price_data, define_direction, delete_intersecting_and_too_small_or_big
from searching_for_definite_objects import get_ticker
from searching_for_definite_objects import mark_wrong_price_results

import sys


def main():
    """The main entry point of the application"""

    img = get_image()
    # if getting the image is unable, to exit the program is necessary
    if img is None:
        sys.exit()

    # in order to fing prices you need to get the border of  the vertical scale
    borders_for_prices = get_borders_of_vertical_scale(img)

    # however, sometimes the found border is wrong and you have to check every border
    nothing_is_found = True
    for border in borders_for_prices:
        # separate area with prices
        img_for_prices = prepare_image_for_price(img, border, 7)

        #  preliminary searching for prices
        red_data = get_price_data(img_for_prices, red_price_info)
        green_data = get_price_data(img_for_prices, green_price_info)
        gray_data = get_price_data(img_for_prices, gray_price_info)
        white_data = get_price_data(img_for_prices, white_price_info)

        # if nothing is found, the border changes to the following
        if len(red_data + green_data + gray_data + white_data) == 0:
            continue
        print("Цены найдены")

        # get ticker
        ticker = get_ticker(img)
        print("Тикер найден")

        # processing of preliminary results
        all_price_results = {
            red_price_key: red_data,
            green_price_key: green_data,
            gray_price_key: gray_data,
            white_price_key: white_data,
        }
        all_price_results = delete_intersecting_and_too_small_or_big(all_price_results)
        all_price_results = mark_wrong_price_results(all_price_results)
        direction = define_direction(all_price_results)
        print("Результаты обработаны")

        # recording
        result_dict = prepare_dict_for_output(all_price_results, direction, ticker)
        print(f"Результаты:\n{result_dict}")
        write_into_json(output_file_path, result_dict)
        print(f"Результаты записаны в файл {output_file_path}")
        nothing_is_found = False
        break

    if nothing_is_found:
        print("Не удалось ничего найти")


if __name__ == '__main__':
    main()
