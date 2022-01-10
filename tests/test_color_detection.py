import unittest

import cv2
import numpy as np

import config as cfg
from color_detection import check_color_proximity
from color_detection import apply_mask
from color_detection import get_mean_color


class TestColorDetection(unittest.TestCase):
    def test_check_color_proximity(self):
        exactly_red_color = cfg.mean_colors['red_price_mean_color']
        exactly_green_color = cfg.mean_colors['green_price_mean_color']
        exactly_gray_color = cfg.mean_colors['gray_price_mean_color']
        exactly_white_color = cfg.mean_colors['white_price_mean_color']
        exactly_blue_color = cfg.mean_colors['blue_price_mean_color']

        self.assertTrue(check_color_proximity('red_price_mean_color', exactly_red_color))
        self.assertTrue(check_color_proximity('green_price_mean_color', exactly_green_color))
        self.assertTrue(check_color_proximity('gray_price_mean_color', exactly_gray_color))
        self.assertTrue(check_color_proximity('white_price_mean_color', exactly_white_color))
        self.assertTrue(check_color_proximity('blue_price_mean_color', exactly_blue_color))

        self.assertFalse(check_color_proximity('red_price_mean_color', exactly_green_color))
        self.assertFalse(check_color_proximity('green_price_mean_color', exactly_red_color))
        self.assertFalse(check_color_proximity('white_price_mean_color', exactly_gray_color))

    def test_apply_mask(self):

        img_hsv = np.ones((1, 1, 3), dtype=np.uint8) * 10
        img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        lower = np.array([0, 0, 0])
        upper = np.array([20, 20, 20])
        img_bgr_filtered = apply_mask(img_bgr, img_hsv, lower, upper)
        self.assertTrue(np.array_equal(img_bgr_filtered, img_bgr))

        lower = np.array([0, 0, 0])
        upper = np.array([5, 5, 5])
        img_bgr_filtered = apply_mask(img_bgr, img_hsv, lower, upper)
        self.assertFalse(np.array_equal(img_bgr_filtered, img_bgr))
        self.assertTrue(np.array_equal(img_bgr_filtered, np.zeros((1, 1, 3), dtype=np.uint8)))

        lower = np.array([0, 0, 0])
        upper = np.array([20, 5, 20])
        img_bgr_filtered = apply_mask(img_bgr, img_hsv, lower, upper)
        self.assertTrue(np.array_equal(img_bgr_filtered, np.zeros((1, 1, 3), dtype=np.uint8)))

    def test_mean_color(self):
        img1 = np.array([[[1, 2, 3]]], dtype=np.uint8)
        img2 = np.zeros((2, 2, 3), dtype=np.uint8)
        img3 = np.ones((2, 2, 3), dtype=np.uint8)
        img4 = np.array([[[2, 2, 2]], [[4, 4, 4]]], dtype=np.uint8)

        mean1 = get_mean_color(img1)
        mean2 = get_mean_color(img2)
        mean3 = get_mean_color(img3)
        mean4 = get_mean_color(img4)

        self.assertTrue(mean1 == [1, 2, 3])
        self.assertTrue(mean2 == [0, 0, 0])
        self.assertTrue(mean3 == [1, 1, 1])
        self.assertTrue(mean4 == [3, 3, 3])


if __name__ == '__main__':
    unittest.main()
