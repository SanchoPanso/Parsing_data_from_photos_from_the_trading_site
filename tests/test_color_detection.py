import unittest
import config as cfg
from color_detection import check_color_proximity


class TestColorDetection(unittest.TestCase):
    def test_check_color_proximity(self):
        exactly_red_color = cfg.mean_colors['red_price_mean_color']
        exactly_green_color = cfg.mean_colors['green_price_mean_color']
        exactly_gray_color = cfg.mean_colors['gray_price_mean_color']
        exactly_white_color = cfg.mean_colors['white_price_mean_color']
        exactly_blue_color = cfg.mean_colors['blue_price_mean_color']

        self.assertTrue(check_color_proximity('red_price_mean_color', exactly_red_color), "red")


if __name__ == '__main__':
    unittest.main()
