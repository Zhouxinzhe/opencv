import cv2
import matplotlib.pyplot as plt
import numpy as np

def cv_show(img, name=' '):
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 透视变换
    # img = cv2.imread('../media/card.jpg')
    # h, w = img.shape[:2]
    # cv_show(img)
    # canny = cv2.Canny(img, 100, 200)
    # cv_show(canny)
    # contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # epsilon = 0.02 * cv2.arcLength(contours[0], True)
    # approx = cv2.approxPolyDP(contours[0], epsilon, True)
    # cv2.drawContours(img, [approx], -1, (0, 255, 0), 1)
    # cv_show(img)
    # pts1 = np.float32(approx)
    # pts2 = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
    # M = cv2.getPerspectiveTransform(pts1, pts2)
    # warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    # cv_show(warped)

# 角点检测
img = cv2.imread('../media/board.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
img_corner = img.copy()
img_corner[dst > 0.05 * dst.max()] = [0, 0, 255]
res = np.hstack((img, img_corner))
cv_show(res)