import cv2


def get_all_approx_contours(img,
                            canny_thresholds: tuple = (30, 60),
                            gauss_kernel: tuple = (7, 7),
                            morph_kernel: tuple = (1, 1)):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, gauss_kernel, 0)

    edged = cv2.Canny(blur, canny_thresholds[0], canny_thresholds[1])

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    con_poly = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            con_poly.append(approx)

    # poly = cv2.drawContours(img, con_poly, -1, (0, 255, 255), 2)
    # cv2.imshow('img', cv2.resize(poly, (640, 640)))
    # cv2.waitKey(0)

    return con_poly


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

