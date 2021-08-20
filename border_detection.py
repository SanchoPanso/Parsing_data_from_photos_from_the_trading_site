import cv2
import numpy as np


class PreprocessingParams:
    def __init__(self,
                 canny_thresholds: tuple,
                 gauss_kernel_size: tuple or None,
                 morph_kernel_size: tuple):

        self.canny_thresholds = canny_thresholds
        self.gauss_kernel_size = gauss_kernel_size
        self.morph_kernel_size = morph_kernel_size


pre_params_for_borders = PreprocessingParams(canny_thresholds=(30, 60),
                                             gauss_kernel_size=(7, 7),
                                             morph_kernel_size=(1, 1))


def preprocessing(img: np.ndarray, params: PreprocessingParams):
    canny_thresholds = params.canny_thresholds
    gauss_kernel_size = params.gauss_kernel_size
    morph_kernel_size = params.morph_kernel_size

    img_result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gauss_kernel_size is not None:
        img_result = cv2.GaussianBlur(img_result, gauss_kernel_size, 0)
    img_result = cv2.Canny(img_result, canny_thresholds[0], canny_thresholds[1])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
    img_result = cv2.morphologyEx(img_result, cv2.MORPH_CLOSE, kernel)

    return img_result


def get_all_approx_contours(img: np.ndarray):

    preprocessed_img = preprocessing(img, pre_params_for_borders)
    contours = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    poly_contours = []
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            poly_contours.append(approx)

    # poly = cv2.drawContours(img, poly_contours, -1, (0, 255, 255), 2)
    # cv2.imshow('img', cv2.resize(poly, (250, 640)))
    # cv2.waitKey(0)

    return poly_contours


def get_bounding_boxes(contours):
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        bboxes.append((x, y, w, h))
    return bboxes


if __name__ == '__main__':
    img = cv2.imread("example.png")
    img = cv2.resize(img, (640, 640))

    cv2.imshow('img', img[320:640, 0:640])
    cv2.waitKey(0)

