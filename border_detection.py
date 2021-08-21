import cv2
import numpy as np
from color_detection import get_filtered_by_colors_image
import os
import matplotlib.pyplot as plt

class PreprocessingParams:
    def __init__(self,
                 canny_thresholds: tuple,
                 gauss_kernel_size: tuple or None,
                 morph_kernel_size: tuple):

        self.canny_thresholds = canny_thresholds
        self.gauss_kernel_size = gauss_kernel_size
        self.morph_kernel_size = morph_kernel_size
        self.thresholding_flag = False


pre_params_for_borders = PreprocessingParams(canny_thresholds=(30, 60),
                                             gauss_kernel_size=(7, 7),
                                             morph_kernel_size=(5, 5))


def preprocessing(img: np.ndarray, params: PreprocessingParams):
    canny_thresholds = params.canny_thresholds
    gauss_kernel_size = params.gauss_kernel_size
    morph_kernel_size = params.morph_kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)

    img_result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gauss_kernel_size is not None:
        img_result = cv2.GaussianBlur(img_result, gauss_kernel_size, 0)

    # img_result = cv2.morphologyEx(img_result, cv2.MORPH_CLOSE, kernel, iterations=3)
    # img_result = cv2.morphologyEx(img_result, cv2.MORPH_OPEN, kernel, iterations=2)

    cv2.imshow('img', cv2.resize(img_result, (160, 720)))
    cv2.waitKey(0)

    img_result = cv2.Canny(img_result, canny_thresholds[0], canny_thresholds[1])
    img_result = cv2.morphologyEx(img_result, cv2.MORPH_CLOSE, kernel)

    return img_result


def get_all_approx_contours(img: np.ndarray, pre_params: PreprocessingParams = pre_params_for_borders):

    preprocessed_img = preprocessing(img, pre_params)
    cv2.imshow('img', cv2.resize(preprocessed_img, (160, 720)))
    cv2.waitKey(0)
    contours = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    poly_contours = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:    # доработать
            poly_contours.append(approx)

    poly = cv2.drawContours(img, poly_contours, -1, (0, 255, 255), 2)
    cv2.imshow('img', cv2.resize(poly, (160, 720)))
    cv2.waitKey(0)

    return poly_contours


def find_contour_with_the_biggest_area(contours):
    max_area = 0
    max_index = -1
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > max_area:
            max_area = cv2.contourArea(contours[i])
            max_index = i
    return contours[max_index]


def get_bounding_boxes(contours):
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, w, h))

    return bboxes


def get_borders_of_vertical_scale(img):
    height = img.shape[0]
    width = img.shape[1]
    filtered = get_filtered_by_colors_image(img, np.array([0, 0, 0]), np.array([180, 85, 115]))
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blur, 50, 150)
    # cv2.imshow("img", edged[:, int(0.9*width):])
    # cv2.waitKey(0)
    borders = []
    sums = []
    for x in range(width - 1, int(0.9*width), -1): # на будущее, переработать
        sum = 0
        for y in range(0, height):
            if edged[y, x] != 0:
                sum += 1
        sums.append(sum/height)
        if sum/height > 0.6:
            if len(borders) > 0 and borders[-1] - x < 5: # 5 как то назвать
                continue
            borders.append(x)

    return borders


if __name__ == '__main__':
    img = cv2.imread(f"example2.jpg")
    print(get_borders_of_vertical_scale(img))
    # img_result = cv2.cvtColor(img[:, int(img.shape[1]*0.9):], cv2.COLOR_BGR2GRAY)
    # img_result = cv2.GaussianBlur(img_result, (3, 3), 0)
    # img_result = cv2.Canny(img_result, 30, 60)
    # cv2.imshow("img", img_result)
    # cv2.waitKey(0)

    # img_paths = os.listdir("test_images")
    # print(img_paths)
    # for path in img_paths:
    #     img = cv2.imread(f"test_images\\{path}")
    #     borders = get_borders_of_vertical_scale(img)
    #     print(borders)


