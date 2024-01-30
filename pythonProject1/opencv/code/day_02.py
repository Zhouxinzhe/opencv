import cv2
import matplotlib.pyplot as plt
import numpy as np

# 边界填充
    # top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
    # img = cv2.imread('../media/scenery.jpg')
    # cv2.namedWindow('constant', cv2.WINDOW_NORMAL)
    # replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REPLICATE)
    # reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT)
    # reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
    # wrap = cv2.copyMakeBorder(img, top_size, bottom_size, right_size, left_size, cv2.BORDER_WRAP)
    # constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_CONSTANT,
    #                               value=[255, 255, 255])
    # cv2.imshow('constant', constant)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# 数值计算
    # cv2.namedWindow('img2', cv2.WINDOW_KEEPRATIO)
    # img2 = img + 10
    # cv2.imshow('img2', img2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# 图像融合
    # cv2.namedWindow('res', cv2.WINDOW_KEEPRATIO)
    # img_sky = cv2.imread('../media/sky.jpg')
    # img_scenery = cv2.imread('../media/scenery.jpg')
    # print(img_sky.shape, img_scenery.shape)
    # img_sky = cv2.resize(img_sky, (1280, 1706))
    # print(img_sky.shape, img_scenery.shape)
    # res = cv2.addWeighted(img_sky, 0.5, img_scenery, 0.5, 0)
    # cv2.imshow('res', res)
    # cv2.waitKey(0)

# 图像阈值
    # img_gray = cv2.cvtColor(img_sky, cv2.COLOR_BGR2GRAY)
    # ret1, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    # ret2, thresh2 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    # ret3, thresh3 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TRUNC)
    # ret4, thresh4 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO)
    # ret5, thresh5 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_TOZERO_INV)
    # titles = ['Original Image', 'Binary Image', 'Binary Inverted Image', 'Trunc Image', 'Tozero Image',
    #           'Tozero Inverted Image']
    # images = [img_gray, thresh1, thresh2, thresh3, thresh4, thresh5]
    # for i in range(6):
    #     plt.subplot(2, 3, i+1)
    #     plt.imshow(images[i], cmap='gray')
    #     plt.title(titles[i])
    #     plt.xticks([])
    #     plt.yticks([])
    #     if i < 5:
    #         ret_value = globals()[f"ret{i + 1}"]
    #         print(f"ret{i + 1}: {ret_value}")  # 打印 ret1 到 ret5 的值
    # plt.show()

# 图像平滑处理
    # img_cartoon = cv2.imread("../media/cartoon.jpg")
    # blur = cv2.blur(img_cartoon, (3, 3))  # 均值滤波
    # box = cv2.boxFilter(img_cartoon, -1, (3, 3), normalize=True)  # 方框滤波
    # gaussian = cv2.GaussianBlur(box, (3, 3), 1)  # 高斯滤波
    # median = cv2.medianBlur(box, 3)  # 中值滤波
    # images = [img_cartoon, blur, box, gaussian, median]
    # titles = ['Original Image', 'Blur Image', 'Box Image', 'Gaussian Image', 'Median Image']
    # for i in range(len(images)):
    #     plt.subplot(2, 3, i + 1)
    #     plt.imshow(images[i])
    #     plt.title(titles[i])
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()

# 腐蚀操作
img = cv2.imread('../media/zhe.jpg')
ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((3, 3), np.uint8)
erosion1 = cv2.erode(img, kernel, iterations=1)
erosion2 = cv2.erode(img, kernel, iterations=2)
res = np.hstack((img, erosion1, erosion2))
cv2.imshow('res', res)
cv2.waitKey(0)

# 膨胀操作
dilation1 = cv2.dilate(erosion2, kernel, iterations=1)
dilation2 = cv2.dilate(erosion2, kernel, iterations=2)
res = np.hstack((erosion2, dilation1, dilation2))
cv2.imshow('res', res)
cv2.waitKey(0)

# 开运算与闭运算
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
images = [img, opening, closing]
titles = ['Original Image', 'Opening Image', 'Closing Image']
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

# 梯度运算
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
images = [img, gradient]
titles = ['Original Image', 'Gradient Image']
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

# 礼帽和黑帽
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
images = [img, tophat, blackhat]
titles = ['Original Image', 'Tophat Image', 'Blackhat Image']
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()