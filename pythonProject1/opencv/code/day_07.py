import cv2
import matplotlib.pyplot as plt
import numpy as np

# SIFT
    # img = cv2.imread('../media/cartoon.jpg.jpg')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # sift = cv2.SIFT.create()
    # kp = sift.detect(gray, None)
    # img = cv2.drawKeypoints(gray, kp, img)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # kp, des = sift.compute(gray, kp)
    # print(des.shape)

# 特征匹配-蛮力匹配
    # img1 = cv2.imread('../media/cartoon.jpg')
    # img2 = cv2.imread('../media/template.jpg')
    # sift = cv2.SIFT.create()
    # kp1, des1 = sift.detectAndCompute(img1, None)
    # kp2, des2 = sift.detectAndCompute(img2, None)
    # bf = cv2.BFMatcher(crossCheck=True)
    # matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    # cv2.imshow('Matches', img3)
    # cv2.waitKey(0)

# test
img = cv2.imread('../media/cartoon.jpg')
sift = cv2.SIFT.create()
kp, des = sift.detectAndCompute(img, None)
print(des.shape, des)