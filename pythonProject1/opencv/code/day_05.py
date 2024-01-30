import cv2
import matplotlib.pyplot as plt
import numpy as np

# 模板匹配
    # img = cv2.imread('../media/cartoon.jpg')
    # template = cv2.imread('../media/template.jpg')
    # img_copy = img.copy()
    # h, w = template.shape[:2:]
    # res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # top_left = max_loc
    # bottom_right = (top_left[0] + w, top_left[1] + h)
    # cv2.rectangle(img_copy, top_left, bottom_right, (0, 0, 255), 1)
    # images = [img, template, img_copy]
    # titles = ['Original Image', 'Template Image', 'Result']
    # for i in range(len(images)):
    #     plt.subplot(1, 3, i + 1), plt.imshow(images[i]), plt.title(titles[i])
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()

# 直方图
    # img = cv2.imread('../media/cartoon.jpg')
    # color = ['b', 'g', 'r']
    # for i, col in enumerate(color):
    #     hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    #     plt.plot(hist, color=col)
    #     plt.xlim([0, 256])
    # plt.show()

# mask
    # img = cv2.imread('../media/cartoon.jpg')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # mask = np.zeros(gray.shape[:2], dtype="uint8")
    # mask[50:100, 50:100] = 255
    # masked_img = cv2.bitwise_and(gray, gray, mask=mask)
    # hist_mask = cv2.calcHist([masked_img], [0], mask, [256], [0, 256])
    # plt.plot(hist_mask)
    # plt.show()

# 均衡化
img = cv2.imread('../media/cartoon.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
equalized_img = cv2.equalizeHist(gray)
plt.subplot(221), plt.imshow(gray, 'gray'), plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(equalized_img, 'gray'), plt.title('Equalized'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.hist(img.ravel(), bins=256, range=(0, 256))
plt.subplot(224), plt.hist(equalized_img.ravel(), bins=256, range=(0, 256))
plt.show()