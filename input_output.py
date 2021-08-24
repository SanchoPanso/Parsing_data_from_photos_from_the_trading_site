from bs4 import BeautifulSoup
import cv2
import json
import numpy as np
import requests
import os
import sys

from config import *
from searching_for_definite_objects import AllPriceResults, PriceResult


def get_image():
    print(sys.argv)
    path = sys.argv[1]
    return get_image_using_path(path)


def get_image_using_path(path: str):
    if not os.path.exists(path):
        print("Файл не найден")
        return None

    image = cv2.imread(path)
    if image is None:
        print("Не удалось открыть файл")
    else:
        print("Файл открыт")
    return image


def get_image_using_url(original_url: str) -> np.ndarray:
    """return the image from the standard page that the url points to"""

    response = requests.get(original_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    img_url = soup.find('img').get('src')

    img_response = requests.get(img_url)
    img_arr = np.asarray(bytearray(img_response.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)

    return img


def prepare_dict_for_output(all_price_results: dict, direction: str, ticker: str):
    result_dict = dict()
    for field in price_data_keys:
        result_dict[field] = []
        for price_result in all_price_results[field]:
            result_dict[field].append(price_result.value)

    result_dict[direction_key] = direction
    result_dict[ticker_key] = ticker

    return result_dict


def write_into_json(filename: str, result_dict: dict):
    with open(filename, "w") as file:
        json.dump(result_dict, file, indent=4)
