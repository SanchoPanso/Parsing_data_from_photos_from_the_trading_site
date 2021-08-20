from bs4 import BeautifulSoup
import cv2
import json
import numpy as np
import requests


def get_image_using_url(original_url: str) -> np.ndarray:
    """return the image from the standard page that the url points to"""

    response = requests.get(original_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    img_url = soup.find('img').get('src')

    img_response = requests.get(img_url)
    img_arr = np.asarray(bytearray(img_response.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)

    return img


def write_into_json(filename: str, values: list, keys: list):
    data_dict = dict()
    for i in range(len(keys)):
        data_dict[keys[i]] = values[i]
    with open(filename, "w") as file:
        json.dump(data_dict, file, indent=4)
