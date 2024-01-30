import cv2
import numpy as np


def cv_show(img, name=' '):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sort_contours(cnts, method='left-to-right'):
    if method == 'left-to-right' or method == 'top-to-bottom':
        rev = False
    elif method == 'right-to-left' or method == 'bottom-to-top':
        rev = True
    if method == 'left-to-right' or method == 'right-to-left':
        i = 0
    elif method == 'top-to-bottom' or method == 'bottom-to-top':
        i = 1
    boundindBoxes = [cv2.boundingRect(cnt) for cnt in cnts]
    (cnts, boundindBoxes) = zip(*sorted(zip(cnts, boundindBoxes), key=lambda b: b[1][i], reverse=rev))
    return cnts, boundindBoxes