import cv2
import numpy as np

###################
# Module is under development and now is not used
##################


# get grayscale image
def get_grayscale():
    def func(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return func


# thresholding
def thresholding():
    def func(img):
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
        for w_left in widths_left:
            for w_right in widths_right:
                return img[:, w_left: img.shape[1] - w_right]
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
preprocessing_for_text_recognition.add(augment((4, 8)))
preprocessing_for_text_recognition.add(thresholding())
preprocessing_for_text_recognition.add(trimming((3, 7, 0), (0,)))

preprocessing_for_border_detection = Preprocessing()
preprocessing_for_border_detection.add(get_grayscale())
preprocessing_for_border_detection.add(closing(range(3, 10, 2)))
preprocessing_for_border_detection.add(paint_borders_black())
preprocessing_for_border_detection.add(gaussian_blur(range(5, 12, 2)))
preprocessing_for_border_detection.add(canny(30, 60))
preprocessing_for_border_detection.add(closing(range(3, 10, 2)))

if __name__ == '__main__':
    pass

