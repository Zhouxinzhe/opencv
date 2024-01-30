import cv2
import matplotlib.pyplot as plt
import numpy as np
import function


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 模板预处理
template = cv2.imread('../images/template.png')
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template_gray = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY)[1]
print(template_gray.shape)
# cv_show(template_gray, 'template_gray')
contours, hierarchy = cv2.findContours(template_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = function.sort_contours(contours, 'left-to-right')[0]
template_draw = cv2.drawContours(template.copy(), contours, -1, (0, 0, 255), 2)
# cv_show(template_draw, 'template_draw')
digits = []
for (i, c) in enumerate(contours):
    x, y, w, h = cv2.boundingRect(c)
    print(x, y, w, h)
    digits.append(template_gray[y:y + h, x:x + w])  # y表示行，x表示列，即第几行第几列进行索引
    digits[i] = cv2.resize(digits[i], (57, 88))
    print(digits[i].shape)
    # cv_show(digits[i], "digits" + str(i))

# 样本处理
img = cv2.imread('../images/credit_card1.jpg')
img = cv2.resize(img, (300, 210))
print(img.shape)
cv_show(img, 'img')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show(img_gray, 'img_gray')
# 礼帽处理，凸显明亮细节
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
tophat = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, kernel)
cv_show(tophat, 'tophat')
# 梯度处理，凸显轮廓
gradx = cv2.Sobel(tophat, cv2.CV_64F, 1, 0, ksize=3)
gradx = cv2.convertScaleAbs(gradx)
grady = cv2.Sobel(tophat, cv2.CV_64F, 0, 1, ksize=3)
grady = cv2.convertScaleAbs(grady)
grad = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
cv_show(grad, 'grad')
# 闭操作，先膨胀后腐蚀，使数字块相连
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 6))
close = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel)
cv_show(close, 'close')
thresh = cv2.threshold(close, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show(thresh, 'thresh')
# 计算轮廓
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
thresh_draw = cv2.drawContours(img.copy(), contours, -1, (0, 0, 255), 1)
cv_show(thresh_draw, 'thresh_draw')
# 筛选轮廓
img_cnt = img.copy()
locs = []
for (i, c) in enumerate(contours):
    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h)
    if 2.8 < ar < 3.2:
        print(ar)
        locs.append((x, y, w, h))
        cv2.rectangle(img_cnt, (x, y), (x + w, y + h), (0, 0, 255))
        cv_show(img[y:y + h, x:x + w], 'img')
cv_show(img_cnt, 'img_cnt')
locs = sorted(locs, key=lambda x: x[0])

# 识别匹配
output = []
for (i, (x, y, w, h)) in enumerate(locs):
    img_tmp = img.copy()[y - 5:y + h + 5, x - 5:x + w + 5]
    img_tmp_gray = cv2.cvtColor(img_tmp, cv2.COLOR_BGR2GRAY)
    img_tmp_gray = cv2.threshold(img_tmp_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv_show(img_tmp_gray, 'img_tmp_gray')
    cnts, hierarchy = cv2.findContours(img_tmp_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = function.sort_contours(cnts)[0]
    output_tmp = []
    for cnt in cnts:
        gx, gy, gw, gh = cv2.boundingRect(cnt)
        cv2.rectangle(img_tmp, (gx, gy), (gx + gw, gy + gh), (0, 0, 255))
        digit_tmp = img_tmp_gray[gy:gy + gh, gx:gx + gw]
        digit_tmp = cv2.resize(digit_tmp, (57, 88))
        # cv_show(digit_tmp, 'digit_tmp')  # 识别对象
        scores = []
        for digit in digits:
            result = cv2.matchTemplate(digit_tmp, digit, cv2.TM_CCOEFF_NORMED)
            score = cv2.minMaxLoc(result)[1]
            scores.append(score)
        output_tmp.append(str(scores.index(max(scores))))
    output.append(output_tmp)
    cv2.rectangle(img, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 0, 255), 1)
    cv2.putText(img, ' '.join(output_tmp), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
print(output)
cv_show(img, 'img')
