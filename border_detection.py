import cv2
import numpy as np
from color_detection import get_filtered_by_colors_image
import os
from preprocessing import Preprocessing
from config import *


def get_all_approx_contours(img: np.ndarray):

    preprocessing_for_border_detection = Preprocessing()
    preprocessing_for_border_detection.add_gray()
    preprocessing_for_border_detection.add_closing(range(3, 10, 2))
    preprocessing_for_border_detection.add_paint_borders_black()
    preprocessing_for_border_detection.add_gaussian_blur(range(5, 12, 2))
    preprocessing_for_border_detection.add_canny(30, 60)
    preprocessing_for_border_detection.add_closing(range(3, 10, 2))

    poly_contours_list = []
    preprocessed_img_list = preprocessing_for_border_detection.preprocess(img)
    for preprocessed_img in preprocessed_img_list:
        contours = list(cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])

        for i in range(len(contours)):
            contours[i] = cv2.convexHull(contours[i])

        poly_contours = []
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:    # доработать
                poly_contours.append(approx)
        poly_contours_list.append(poly_contours)

        if debug_mode:
            poly = cv2.drawContours(img.copy(), poly_contours, -1, (0, 255, 255), 2)
            cv2.imshow('img', cv2.resize(poly, (160, 720)))
            cv2.waitKey(dbg_delay)

    return poly_contours_list


def find_contour_with_the_biggest_area(contours):
    max_area = 0
    max_index = -1
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > max_area:
            max_area = cv2.contourArea(contours[i])
            max_index = i
    return contours[max_index]


def get_bounding_boxes(contours_list):
    bboxes_list = []
    for contours in contours_list:
        bboxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            bboxes.append((x, y, w, h))
        bboxes_list.append(bboxes)

    return bboxes_list


# def get_borders(img: np.ndarray,
#                 min_dist_from_edge: int,
#                 min_dist_between_borders: int,
#                 prepr: Preprocessing = prepr_for_ticker_border,
#                 borders_is_vertical: bool = True,
#                 start_from_zero: bool = True):
#
#     prepr_img = prepr.preprocess(img)
#     if not borders_is_vertical:
#         prepr_img = prepr_img.T
#     height = img.shape[0]
#     width = img.shape[1]
#
#     borders = []
#     x_range = range(width) if start_from_zero else range(width - 1, width, -1)
#     for x in x_range:
#         sum_of_edged = 0
#         for y in range(0, height):
#             if prepr_img[y, x] != 0:
#                 sum_of_edged += 1
#         if sum_of_edged/height > 0.45:
#             if len(borders) > 0:
#                 if borders[-1] - x < min_dist_between_borders:
#                     continue
#             else:
#                 if width - 1 - x < min_dist_from_edge:
#                     continue
#             borders.append(x)


def get_ticker_borders(img: np.ndarray,
                       min_dist_from_edge: int,
                       min_dist_between_borders: int,
                       ):
    height = img.shape[0]
    width = img.shape[1]

    prepr_for_ticker_border = Preprocessing()
    prepr_for_ticker_border.add_gray()
    prepr_for_ticker_border.add_gaussian_blur(3)
    prepr_for_ticker_border.add_canny(30, 60)

    prepr_img = prepr_for_ticker_border.preprocess(img)[0]
    borders = []

    for y in range(height):
        if y == height - 1:
            borders.append(y)
            break
        sum_of_edged = 0
        for x in range(0, width):
            if prepr_img[y, x] != 0:
                sum_of_edged += 1
        if sum_of_edged/width > 0.45:
            if len(borders) > 0:
                if borders[-1] - y < min_dist_between_borders:
                    continue
            else:
                if width - 1 - y < min_dist_from_edge:
                    continue
            borders.append(y)
    return borders


def get_borders_of_vertical_scale(img):
    height = img.shape[0]
    width = img.shape[1]

    filtered = get_filtered_by_colors_image(img, np.array([0, 0, 0]), np.array([180, 85, 115]))
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blur, 30, 60)

    # cv2.imshow("Edged", edged[:, int(0.9*width):])
    # cv2.waitKey(1000)
    borders = []
    sums = []
    for x in range(width - 1, int(0.9 * width), -1):
        sum = 0
        for y in range(0, height):
            if edged[y, x] != 0:
                sum += 1
        sums.append(sum/height)
        if sum/height > 0.45:
            if len(borders) > 0:
                if borders[-1] - x < 5:
                    continue
            else:
                if width - 1 - x < 17:
                    continue
            borders.append(x)
    # mean = np.array(sums).mean()
    # print(mean)
    # means = [mean] * len(sums)
    # plt.plot(sums)
    # plt.plot(means)
    # plt.show()

    return borders


if __name__ == '__main__':
    pass

