import numpy as np
import cv2
import pytesseract
from pytesseract import Output
import re
import os

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def delete_black_borders(img):
    mean = [0, 0, 0]
    width = img.shape[1]
    height = img.shape[0]
    depth = 0
    black_borders_is_not_deleted = True
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    threshold = 140

    for x in range(width):
        for y in range(height):
            for canal in range(3):
                mean[canal] += img[y, x][canal]
    for canal in range(3):
        mean[canal] /= width * height

    while black_borders_is_not_deleted:
        black_borders_is_not_deleted = False
        for x in range(depth, width - depth):
            y = 0
            if img_hsv[y, x][2] < threshold:
                black_borders_is_not_deleted = True
                for canal in range(3):
                    img[y, x][canal] = mean[canal]

        for x in range(depth, width - depth):
            y = height - 1
            if img_hsv[y, x][2] < threshold:
                black_borders_is_not_deleted = True
                for canal in range(3):
                    img[y, x][canal] = mean[canal]

        for y in range(depth, height - depth):
            x = 0
            if img_hsv[y, x][2] < threshold:
                black_borders_is_not_deleted = True
                for canal in range(3):
                    img[y, x][canal] = mean[canal]

        for y in range(depth, height - depth):
            x = width - 1
            if img_hsv[y, x][2] < threshold:
                black_borders_is_not_deleted = True
                for canal in range(3):
                    img[y, x][canal] = mean[canal]
        depth += 1
    return img


def preprocessing_for_text_recognition(img, aug_value=4):
    fixed = delete_black_borders(img)
    gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
    augmented = cv2.resize(gray, (gray.shape[1] * 4,
                                  gray.shape[0] * 4))
    thresholded = cv2.threshold(augmented, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresholded


def get_text_data(img):
    try:
        text_data = pytesseract.image_to_data(img, lang='eng', output_type=Output.DICT,
                                              config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789.:')
        return text_data
    except Exception:
        return {'text': []}


def get_text(img):
    text = pytesseract.image_to_string(img)
    return text


def highlight_text_on_image(img):
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    print(d['text'])
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    img = cv2.imread("example2.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    opened = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel)
    img = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
    # threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    highlight_text_on_image(img[:, 2520:2556])
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #
    # print(pytesseract.image_to_string(img, lang='eng',
    #                                   config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789.'))


    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edged = cv2.Canny(gray, 30, 60)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    # closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("aaaa", cv2.resize(closed, (720, 720)))
    # cv2.waitKey(0)

