import cv2
import numpy as np
from color_detection import get_filtered_by_colors_image
import os
import matplotlib.pyplot as plt
from preprocessing import preprocessing_for_border_detection


def get_all_approx_contours(img: np.ndarray):
    poly_contours_list = []
    preprocessed_img_list = preprocessing_for_border_detection.preprocess(img)
    for preprocessed_img in preprocessed_img_list:
        contours = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for i in range(len(contours)):
            contours[i] = cv2.convexHull(contours[i])

        poly_contours = []
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:    # доработать
                poly_contours.append(approx)
        poly_contours_list.append(poly_contours)

        # poly = cv2.drawContours(img.copy(), poly_contours, -1, (0, 255, 255), 2)
        # cv2.imshow('img', cv2.resize(poly, (160, 720)))
        # cv2.waitKey(1)

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
                if borders[-1] - x < 10: # 5 как то назвать
                    continue
            else:
                if width - 1 - x < 15:
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

