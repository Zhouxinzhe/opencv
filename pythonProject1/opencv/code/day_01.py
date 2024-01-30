import cv2 # opencv读取的格式是BGR
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Image

img = cv2.imread('../media/scenery.jpg', cv2.IMREAD_COLOR)
print(img)
print(img.dtype)
print(img.shape)

# 图像显示，可以创建多个窗口
cv2.imshow('scenery', img)
# 等待时间，单位毫秒，0表示任意键终止
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存图像
cv2.imwrite('../media/myscenery.png', img)

vc = cv2.VideoCapture('../media/video.mp4')
# 检查是否打开正确
if vc.isOpened():
    isopen, Frame = vc.read()
else:
    isopen = False

while isopen:
    ret, Frame = vc.read()
    if Frame is None:
        break
    if ret:
        gray = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)
        if cv2.waitKey(10) & 0xFF == 27:
            break
vc.release()
cv2.destroyAllWindows()