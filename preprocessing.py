import cv2
import numpy as np

###################
# Module is under development and now is not used
##################

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
            img_result = function(img_result)
        return img_result


# get grayscale image
def get_grayscale(img):
    def func(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return func

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)



