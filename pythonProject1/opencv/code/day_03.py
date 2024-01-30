import cv2
import matplotlib.pyplot as plt
import numpy as np

# sobel
    # img = cv2.imread('../media/zhe.jpg')
    # ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    # sobelx = cv2.convertScaleAbs(sobelx)
    # sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    # sobely = cv2.convertScaleAbs(sobely)
    # sobelxy = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
    # images = [img, sobelx, sobely, sobelxy]
    # titles = ['Original Image', 'Sobelx Image', 'Sobely Image', 'Sobelxy Image']
    # for i in range(len(images)):
    #     plt.subplot(2, 2, i + 1)
    #     plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()

# scharr
img = cv2.imread('../media/apple.jpg', cv2.IMREAD_GRAYSCALE)
Scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
Scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
Scharrx = cv2.convertScaleAbs(Scharrx)
Scharry = cv2.convertScaleAbs(Scharry)
Scharr = cv2.addWeighted(Scharrx, 0.5, Scharry, 0.5, 0)
median = cv2.medianBlur(Scharr, 3)
images = [img, Scharr, median]
titles = ['Original Image', 'Scharr Image', 'Median Image']
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

# laplacian
    # img = cv2.imread('../media/apple.jpg', cv2.IMREAD_GRAYSCALE)
    # laplacian = cv2.Laplacian(img, cv2.CV_64F)
    # laplacian = cv2.convertScaleAbs(laplacian)
    # images = [img, laplacian]
    # titles = ['Original Image', 'Laplacian Image']
    # for i in range(len(images)):
    #     plt.subplot(1, 2, i + 1)
    #     plt.imshow(images[i], cmap='gray')
    #     plt.title(titles[i])
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()