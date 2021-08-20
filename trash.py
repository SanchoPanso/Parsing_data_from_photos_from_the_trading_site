import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup


def get_image_using_url(original_url: str) -> np.ndarray:
    """return the image from the standard page that the url points to"""

    response = requests.get(original_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    img_url = soup.find('img').get('src')

    img_response = requests.get(img_url)
    img_arr = np.asarray(bytearray(img_response.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)

    return img

def trash():
    # cv2.imwrite("example.png", img)

    img = cv2.imread("example.png")

    img = cv2.resize(img, (640, 640))

    img = cv2.GaussianBlur(img, (1, 1), 0)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower = np.array([0, 195, 0])# for red
    # upper = np.array([23, 255, 255])

    lower = np.array([85, 0, 135])# for green
    upper = np.array([98, 255, 255])

    # lower = np.array([103, 0, 108]) # for gray
    # upper = np.array([122, 66, 148])

    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('img', imgResult)
    cv2.waitKey(0)

    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # d = pytesseract.image_to_data(imgResult, output_type=Output.DICT)
    # print(d['text'])
    #
    # n_boxes = len(d['text'])
    # for i in range(n_boxes):
    #     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #     img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    gray = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blur, 30, 60)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('img', closed)
    cv2.waitKey(0)

    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    con_poly = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            con_poly.append(approx)

    # black = np.zeros((640, 640, 3), np.uint8)
    poly = cv2.drawContours(img, con_poly, -1, (0, 255, 255), 2)

    cv2.imshow('img', poly)
    cv2.waitKey(0)

# img = cv2.imread("example.png")
    # filtered_img = get_filtered_by_colors_image(img, green_price_lower, green_price_upper)
    # filtered_img = preprocessing(filtered_img)
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # start = time.time()
    # d = pytesseract.image_to_boxes(filtered_img, output_type=Output.DICT)
    # print(d.keys())
    # print(time.time() - start)
    # n_boxes = len(d['char'])
    # print(n_boxes)
    # for i in range(n_boxes):
    #     (x, y, w, h) = (d['left'][i], d['top'][i], d['right'][i], d['bottom'][i])
    #     img = cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
    # cv2.imshow('img', cv2.resize(img, (640, 640)))
    # cv2.waitKey(0)