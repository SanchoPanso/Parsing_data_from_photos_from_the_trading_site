import numpy as np
import cv2
import pytesseract
from pytesseract import Output
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def get_text_data(img):
    try:
        text_data = pytesseract.image_to_data(img, output_type=Output.DICT)
        return text_data
    except Exception:
        return {'text': []}


def get_text(img):
    text = pytesseract.image_to_string(img)
    return text


if __name__ == '__main__':
    img = cv2.imread("example.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("aaaa", cv2.resize(closed, (720, 720)))
    cv2.waitKey(0)

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    d = pytesseract.image_to_data(closed, output_type=Output.DICT)
    print(d['text'])
    #
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if re.search(r"\d{1,10}[.]\d{1,10}", d['text'][i]):
            print(d['text'][i])
