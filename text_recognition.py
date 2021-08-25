import numpy as np
import cv2
import pytesseract
from pytesseract import Output
import re
import os

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class TextCash:
    def __init__(self, threshold=30, max_n_marks=2):
        self.bboxes = []
        self.text = []
        self.marks = []
        self.threshold = threshold
        self.max_n_marks = max_n_marks

    def check(self, query_bbox):
        index = -1
        for i in range(len(self.bboxes)):
            cash_bbox = self.bboxes[i]
            if ((np.array(cash_bbox) - np.array(query_bbox))**2).sum() < self.threshold:
                if len(self.text[i]) > 0:
                    index = i
                elif index < 0 and self.marks[i] < self.max_n_marks:
                    index = -2
        return index

    def add(self, bbox, text):
        self.bboxes.append(bbox)
        self.text.append(text)
        self.marks.append(0)

    def upsert(self, bbox, new_text):
        flag = True
        for i in range(len(self.bboxes)):
            cash_bbox = self.bboxes[i]
            if ((np.array(cash_bbox) - np.array(bbox)) ** 2).sum() < self.threshold:
                if len(new_text) > len(self.text[i]):
                    # print(new_text)
                    self.text[i] = new_text
                    self.marks[i] = 0
                else:
                    self.marks[i] += 1
        if flag:
            print(new_text)
            self.bboxes.append(bbox)
            self.text.append(new_text)
            self.marks.append(0)

# продумать обработку мест с долго не распознающимся текстом


def get_digit_only_text_data(img):
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
    img = cv2.imread("images_for_experiments\\current_price_snippet.jpg")
    print(get_digit_only_text_data(img))


