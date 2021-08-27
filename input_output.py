from bs4 import BeautifulSoup
import cv2
import json
import numpy as np
import requests
import os
import sys
import argparse

from config import *

example_url = "https://www.tradingview.com/x/nShwrpHU/"
example_path = "example.jpg"


def get_image():
    """get image using command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('path_or_url', nargs='?', default=example_path)
    parser.add_argument('-u', '--url', action='store_true', default=False)
    namespace = parser.parse_args(sys.argv[1:])
    if namespace.url:
        url = namespace.path_or_url
        return get_image_using_url(url)
    else:
        path = namespace.path_or_url
        return get_image_using_path(path)


def get_image_using_path(path: str):
    """get image using paths"""
    if not os.path.exists(path):
        print("Файл не найден")
        return None

    image = cv2.imread(path)
    if image is None:
        print("Не удалось открыть файл")
    else:
        print("Файл открыт")
    return image


def get_image_using_url(original_url: str) -> np.ndarray or None:
    """return the image from the standard page that the url points to"""
    try:
        response = requests.get(original_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        img_url = soup.find('img').get('src')

        img_response = requests.get(img_url)
        img_arr = np.asarray(bytearray(img_response.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        if img is None:
            print("Не удалось открыть файл")
        else:
            print("Файл открыт")
        return img
    except Exception as e:
        print(e)
        print("Не удалось открыть файл")
        return None


def prepare_dict_for_output(all_price_results: dict, direction: str, ticker: str):
    """collect prices, direction, ticker into a single dictionary"""
    result_dict = dict()
    for field in price_data_keys:
        result_dict[field] = []
        for price_result in all_price_results[field]:
            result_dict[field].append(price_result.value)

    result_dict[direction_key] = direction
    result_dict[ticker_key] = ticker

    return result_dict


def write_into_json(filename: str, result_dict: dict):
    """write result dictionary into a json file"""
    with open(filename, "w") as file:
        json.dump(result_dict, file, indent=4)


if __name__ == '__main__':
    cv2.imshow('example_url', get_image_using_url(example_url))
    cv2.imshow('example_path', get_image_using_path(example_path))
    if cv2.waitKey(0) == 27:
        sys.exit()
