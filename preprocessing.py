import cv2
import numpy as np


# get grayscale image
def get_grayscale():
    def func(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return func


# thresholding
def thresholding(inv=False):
    def func(img):
        sq = img.shape[0]*img.shape[1]
        if inv:
            mn = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1].sum() / sq
            if mn > 150:
                return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            if mn < 100:
                return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            return [cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                    cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]]
        else:
            return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return func


# gaussian blur
def gaussian_blur(size: int or range or tuple):
    def func(img):
        if type(size) == int:
            return cv2.GaussianBlur(img, (size, size), 0)
        elif type(size) in [range, tuple]:
            result = []
            for s in size:
                result.append(cv2.GaussianBlur(img, (s, s), 0))
        return result
    return func


# canny edge detection
def canny(thr1, thr2):
    def func(img):
        return cv2.Canny(img, thr1, thr2)
    return func


# closing - dilation followed by erosion
def closing(size: int or range or tuple, iterations: int = 1):
    def func(img):
        if type(size) == int:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif type(size) in [range, tuple]:
            result = []
            for s in size:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (s, s))
                result.append(cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations))
        return result
    return func


# opening - erosion followed by dilation
def opening(size: int or range or tuple, iterations=1):
    def func(img):
        if type(size) == int:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif type(size) in [range, tuple]:
            result = []
            for s in size:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (s, s))
                result.append(cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations))
        return result
    return func


# dilation
def dilate(size: int or range or tuple, iterations=1):
    def func(img):
        if type(size) == int:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            return cv2.dilate(img, kernel, iterations=iterations)
        elif type(size) in [range, tuple]:
            result = []
            for s in size:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (s, s))
                result.append(cv2.dilate(img, kernel, iterations=iterations))
        return result
    return func


# erosion
def erode(size: int or range or tuple, iterations=1):
    def func(img):
        if type(size) == int:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
            return cv2.erode(img, kernel, iterations=iterations)
        elif type(size) in [range, tuple]:
            result = []
            for s in size:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (s, s))
                result.append(cv2.erode(img, kernel, iterations=iterations))
        return result
    return func


# augmentation
def augment(aug_value: int or range or tuple):
    def func(img):
        if type(aug_value) == int:
            return cv2.resize(img, None, fx=aug_value, fy=aug_value)
        elif type(aug_value) in [range, tuple]:
            result = []
            for i in aug_value:
                result.append(cv2.resize(img, None, fx=i, fy=i))
        return result
    return func


def trimming(widths_left, widths_right):
    def func(img):
        original_width = img.shape[1]
        original_height = img.shape[0]
        result = []
        for w_left in widths_left:
            for w_right in widths_right:
                im = img.copy()
                im = im[:, int(w_left * original_height): original_width - int(w_right * original_height)]
                result.append(im)
        return result
    return func


def paint_borders_black():
    def func(img):
        x1 = 0
        x2 = img.shape[1] - 1
        for y in range(img.shape[0]):
            img[y, x1] = 0
            img[y, x2] = 0
        return img
    return func


class Preprocessing:
    """
    It is a class for preprocessing implementation. The instance of this class have specific settings
    for specific image preprocessing.
    """
    def __init__(self):
        """initialization"""
        self.sequencing = []

    def add(self, function):
        """add the function to sequencing"""
        self.sequencing.append(function)

    def clean(self):
        """make the sequencing empty"""
        self.sequencing = []

    def preprocess(self, img):
        """preprocess image with settings, which was chosen previously"""
        img_result = img.copy()
        for function in self.sequencing:
            if type(img_result) == list:
                intermediate_result = []
                for i in range(len(img_result) - 1, -1, -1):
                    current_result = function(img_result[i])
                    if type(current_result) == list:
                        intermediate_result += current_result
                    else:
                        intermediate_result.append(current_result)
                img_result = intermediate_result
            else:
                img_result = function(img_result)
        return img_result


preprocessing_for_text_recognition = Preprocessing()
preprocessing_for_text_recognition.add(get_grayscale())
preprocessing_for_text_recognition.add(augment((4, 5.5, 8)))
preprocessing_for_text_recognition.add(thresholding(True))
preprocessing_for_text_recognition.add(trimming((0, 0.25), (0,)))

preprocessing_for_border_detection = Preprocessing()
preprocessing_for_border_detection.add(get_grayscale())
preprocessing_for_border_detection.add(closing(range(3, 10, 2)))
preprocessing_for_border_detection.add(paint_borders_black())
preprocessing_for_border_detection.add(gaussian_blur(range(5, 12, 2)))
preprocessing_for_border_detection.add(canny(30, 60))
preprocessing_for_border_detection.add(closing(range(3, 10, 2)))

prepr_for_ticker_border = Preprocessing()
prepr_for_ticker_border.add(get_grayscale())
prepr_for_ticker_border.add(gaussian_blur(3))
prepr_for_ticker_border.add(canny(30, 60))

if __name__ == '__main__':
    pass

