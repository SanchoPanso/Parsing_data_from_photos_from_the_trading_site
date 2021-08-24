import time
import os

from config import *
from border_detection import get_borders_of_vertical_scale

from input_output import write_into_json, prepare_dict_for_output
from input_output import get_image, get_image_using_path

from searching_for_definite_objects import PriceResult
from searching_for_definite_objects import red_price_info, green_price_info, gray_price_info, white_price_info
from searching_for_definite_objects import prepare_image_for_price
from searching_for_definite_objects import get_price_data, define_direction, delete_intersecting_and_small
from searching_for_definite_objects import get_ticker
from searching_for_definite_objects import mark_wrong_price_results

from debug_and_testing import highlight_prices


def main():
    """The main entry point of the application"""

    # print(sys.argv)
    start_time = time.time()

    img = get_image_using_path("test_images\\f_022611fa253a07ec.jpg")

    borders_for_prices = get_borders_of_vertical_scale(img)
    img_for_prices = prepare_image_for_price(img[:, borders_for_prices[0]-7: img.shape[1]], 9)

    ticker = get_ticker(img)
    print(f"Тикер: {get_ticker(img)}")

    red_data = get_price_data(img_for_prices, red_price_info)
    print(f"Красная цена: {[str(elem) for elem in red_data]}")

    green_data = get_price_data(img_for_prices, green_price_info)
    print(f"Зеленая цена: {[str(elem) for elem in green_data]}")

    gray_data = get_price_data(img_for_prices, gray_price_info)
    print(f"Серая цена: {[str(elem) for elem in gray_data]}")

    white_data = get_price_data(img_for_prices, white_price_info)
    print(f"Белая цена: {[str(elem) for elem in white_data]}")

    all_price_results = {
        red_price_key: red_data,
        green_price_key: green_data,
        gray_price_key: gray_data,
        white_price_key: white_data,
    }

    all_price_results = delete_intersecting_and_small(all_price_results)
    all_price_results = mark_wrong_price_results(all_price_results)
    direction = define_direction(all_price_results)

    print("Время работы: {:.2f} с".format(time.time() - start_time))
    # highlight_prices(img, borders_for_prices[0], all_price_results, ticker, direction, 0)

    result_dict = prepare_dict_for_output(all_price_results, direction, ticker)
    write_into_json(output_file_path, result_dict)


if __name__ == '__main__':
    main()

