import numpy as np
import cv2
import re
import time

from color_detection import get_filtered_by_colors_image
from border_detection import get_all_approx_contours, get_bounding_boxes
from text_detection import get_text_data, get_text

example_url = "https://www.tradingview.com/x/nShwrpHU/"

red_price_lower = [0, 0, 0]
red_price_upper = [40, 255, 255]

green_price_lower = [85, 0, 125]
green_price_upper = [90, 255, 255]

gray_price_lower = [114, 15, 0]
gray_price_upper = [116, 40, 138]

red_area_lower = [143, 0, 0]
red_area_upper = [163, 255, 255]

green_area_lower = [79, 0, 0]
green_area_upper = [92, 106, 61]


def preprocessing(img: np.ndarray) -> np.ndarray:
    """use some filters to transform an image for better text recognition"""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    return closed


def get_data_from_boxes(img: np.ndarray, lower: list, upper: list, patterns: tuple = (r"\d{1,}[.]\d{1,}",)) -> list:
    """
    filter image by colors and then find rectangles in this image,
    find their bounding boxes and recognize text in boxes which fulfill the required patterns
    """

    filtered_img = get_filtered_by_colors_image(img, lower, upper)
    approx_contours = get_all_approx_contours(filtered_img)
    bboxes = get_bounding_boxes(approx_contours)

    valid_text = []
    for bbox in bboxes:
        x, y, w, h = bbox
        if 1 <= w / h <= 4:
            cropped_img = img[y:y + h, x:x + w]
            preprocessed_img = preprocessing(cropped_img)
            text = get_text_data(preprocessed_img)['text']
            valid_words = []
            for word in text:
                for pattern in patterns:
                    if re.search(pattern, word):
                        valid_words.append(word)
            if len(valid_words) > 0:
                valid_text.append(valid_words)
    for elem in valid_text:
        if len(elem) == 1:
            return elem[0]
    return None


def get_current_area(img):
    """define in which area the current price is"""
    red_areas_img = get_filtered_by_colors_image(img, red_area_lower, red_area_upper)
    green_areas_img = get_filtered_by_colors_image(img, green_area_lower, green_area_upper)
    areas_img = red_areas_img + green_areas_img
    no_area_is_reached = True
    coord_x = img.shape[1] - 1
    coord_y = img.shape[0] // 2
    while no_area_is_reached:
        if img[coord_y, coord_x][0] != 0 and img[coord_y, coord_x][1] != 0 and img[coord_y, coord_x][2] != 0:
            if img[coord_y, coord_x][1] > img[coord_y, coord_x][2]:
                return "green"
            else:
                return "red"
        coord_x -= 1


def get_title(img):
    img_for_title = img[:80, :]
    text = get_text(img_for_title)
    title = text.split('\n')[1].split(',')[0]
    return title


def main():
    """The main entry point of the application"""

    start_time = time.time()
    img = cv2.imread("example.png")
    img_for_prices = img[:, int(0.9 * img.shape[1]):img.shape[1]]   # доработать

    print(f"Пара: {get_title(img)}")
    print(f"Текущая область: {get_current_area(img)}")

    gray_data = get_data_from_boxes(img_for_prices, gray_price_lower, gray_price_upper)
    print(f"Серая цена: {gray_data}")

    red_data = get_data_from_boxes(img_for_prices, red_price_lower, red_price_upper)
    print(f"Красная цена: {red_data}")

    green_data = get_data_from_boxes(img_for_prices, green_price_lower, green_price_upper, (r"\d{1,}[.]\d{1,}",
                                                                                            r"\d{2}:\d{2}:\d{2}"))
    print(f"Зеленая цена: {green_data}")

    print(f"Время работы: {time.time() - start_time} с")


if __name__ == '__main__':
    main()


