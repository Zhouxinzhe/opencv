import cv2
import matplotlib.pyplot as plt
import numpy as np

# Canny边缘检测
    # img = cv2.imread('../media/apple.jpg', cv2.IMREAD_GRAYSCALE)
    # v1 = cv2.Canny(img, 100, 150)
    # images = [img, v1]
    # titles = ['Original Image', 'Canny Image']
    # for i in range(len(images)):
    #     plt.subplot(1, 2, i + 1)
    #     plt.imshow(images[i], cmap='gray')
    #     plt.title(titles[i])
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()

# 图像轮廓
img = cv2.imread('../media/car.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_copy = img.copy()
cv2.drawContours(img_copy, contours, -1, (0, 0, 255), 1)
res = np.hstack((img, img_copy))
cv2.imshow('res', res)
cv2.waitKey(0)