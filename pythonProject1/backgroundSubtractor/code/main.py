import cv2
import numpy as np

capture = cv2.VideoCapture('../videos/vtest.avi')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# 创建混合高斯模型用于背景建模
fgbg = cv2.createBackgroundSubtractorMOG2()

if not capture.isOpened():
    print('Unable to open: vtest.av')
    exit(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break
    # 背景建模和前景检测
    fgmask = fgbg.apply(frame)
    # 开运算，去除噪点
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # 轮廓检测
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        if perimeter >= 188:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    # 结果显示
    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)
    k = cv2.waitKey(50) & 0xFF
    if k == 27:
        break
capture.release()
cv2.destroyAllWindows()
