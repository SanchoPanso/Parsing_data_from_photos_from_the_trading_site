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


def preprocessing_for_border_detection(img: np.ndarray,
                                       gauss_kernel_size,
                                       morph_kernel_size1,
                                       morph_kernel_size2,
                                       canny_thresholds=(30, 60)):

    img_result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size1, morph_kernel_size1))
    img_result = cv2.morphologyEx(img_result, cv2.MORPH_CLOSE, kernel1, iterations=1)

    x1 = 0
    x2 = img_result.shape[1] - 1
    for y in range(img_result.shape[0]):
        img_result[y, x1] = 0
        img_result[y, x2] = 0

    img_result = cv2.GaussianBlur(img_result, (gauss_kernel_size, gauss_kernel_size), 0)
    img_result = cv2.Canny(img_result, canny_thresholds[0], canny_thresholds[1])
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size2, morph_kernel_size2))
    img_result = cv2.morphologyEx(img_result, cv2.MORPH_CLOSE, kernel2)

    return img_result


def get_all_approx_contours(img: np.ndarray,
                            gauss_kernel_sizes=range(5, 12, 2),
                            morph_kernel_sizes1=range(3, 10, 2),
                            morph_kernel_sizes2=range(3, 10, 2)):
    poly_contours_list = []
    for gauss_kernel_size in gauss_kernel_sizes:
        for morph_kernel_size1 in morph_kernel_sizes1:
            for morph_kernel_size2 in morph_kernel_sizes2:
                preprocessed_img = preprocessing_for_border_detection(img,
                                                                      gauss_kernel_size=gauss_kernel_size,
                                                                      morph_kernel_size1=morph_kernel_size1,
                                                                      morph_kernel_size2=morph_kernel_size2)

                # cv2.imshow('img', cv2.resize(preprocessed_img, (160, 720)))
                # cv2.waitKey(0)
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

                poly = cv2.drawContours(img.copy(), poly_contours, -1, (0, 255, 255), 2)
                cv2.imshow('img', cv2.resize(poly, (160, 720)))
                cv2.waitKey(1)

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
    for x in range(width - 1, int(0.9 * width), -1): # на будущее, переработать
        sum = 0
        for y in range(0, height):
            if edged[y, x] != 0:
                sum += 1
        sums.append(sum/height)
        if sum/height > 0.45:
            if len(borders) > 0:
                if borders[-1] - x < 5: # 5 как то назвать
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
    img = cv2.imread(f"example.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # cv2.imwrite("images_for_experiments\\current_price_snippet.jpg", cv2.resize(img, None, fx=4, fy=4))
    borders = get_borders_of_vertical_scale(img)
    print(borders)
    cv2.imshow("Borders", img[:, borders[0]:img.shape[1]])
    cv2.waitKey(0)

    # img_paths = os.listdir("test_images")
    # print(img_paths)
    # for path in img_paths:
    #     img = cv2.imread(f"test_images\\{path}")
    #     borders = get_borders_of_vertical_scale(img)
    #     print(borders)
    #     cv2.imshow("Borders", img[:, borders[0]:img.shape[1]])
    #     cv2.waitKey(200)

