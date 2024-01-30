import cv2
import numpy as np
import function as f

question_num = 5
option_num = 5
correct_ans = [2, 5, 1, 3, 4]
input_ans = []
score = 0

img = cv2.imread('../images/paper.jpg')
# 灰度化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
f.cv_show(gray)
# 边缘检测
canny = cv2.Canny(gray, 50, 150)
f.cv_show(canny)
# 轮廓检测
contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
# 轮廓近似
arclength = cv2.arcLength(contours[0], True)
approx = cv2.approxPolyDP(contours[0], 0.01 * arclength, True)
img_contours = cv2.drawContours(img.copy(), [approx], -1, (0, 0, 255), 2)
f.cv_show(img_contours)
# 透视变换
pts1 = np.float32(approx)
pts2 = np.float32([[0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]])
M = cv2.getPerspectiveTransform(pts1, pts2)
img_warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
f.cv_show(img_warped)

# 灰度化 & 二值化
img_warped_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
img_warped_thresh = cv2.threshold(img_warped_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
f.cv_show(img_warped_thresh)
# 边缘检测
img_warped_canny = cv2.Canny(img_warped_thresh, 50, 150)
f.cv_show(img_warped_canny)
# 轮廓检测
contours, hierarchy = cv2.findContours(img_warped_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 轮廓筛选
cnts = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 1800 < area <= 2000:  # 根据实际情况确定筛选条件
        cnts.append(cnt)
img_warped_contours = cv2.drawContours(img_warped.copy(), cnts, -1, (0, 0, 255), 2)
f.cv_show(img_warped_contours)
cnts = f.sort_contours(cnts, 'top-to-bottom')[0]
# 判断
img_output = img_warped.copy()
for i in range(question_num):
    cnt = cnts[option_num*i:option_num*i+option_num]
    cnt = f.sort_contours(cnt, 'left-to-right')[0]
    img_output = cv2.drawContours(img_output, cnt, correct_ans[i] - 1, (0, 0, 255), 2)
    for j in range(option_num):
        mask = np.zeros(img_warped_thresh.shape, dtype='uint8')
        mask = cv2.drawContours(mask, cnt, j, (255, 255, 255), -1)  # -1表示填充
        mask = cv2.bitwise_and(img_warped_thresh, img_warped_thresh, mask=mask)
        total = cv2.countNonZero(mask)
        # f.cv_show(mask)
        if total < 1000:
            input_ans.append(j+1)
# 计算分数，输出结果
for i in range(question_num):
    if input_ans[i] == correct_ans[i]:
        score += 1
score = 100 * float(score) / question_num
img_output = cv2.putText(img_output, str(score), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
f.cv_show(img_output)