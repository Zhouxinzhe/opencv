import cv2
import numpy as np

capture = cv2.VideoCapture('../videos/vtest.avi')

# 角点检测需要的参数
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7)

# lucas kanade参数
lk_params = dict(winSize=(15, 15), maxLevel=2)

# random color
colors = np.random.uniform(0, 255, size=(100, 3))

# the first picture
ret, old_frame = capture.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_RGB2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# mask
mask = np.zeros_like(old_frame)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break
    frame_fray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_fray, p0, None, **lk_params)
    good_new = p1[st == 1]
    print(good_new)
    good_old = p0[st == 1]
    # 绘制轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), colors[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, colors[i].tolist(), -1)
    img = cv2.add(mask, frame)
    cv2.imshow('img', img)
    k = cv2.waitKey(100) & 0xFF
    if k == 27:
        break

    old_gray = frame_fray.copy()
    p0 = good_new.reshape(-1, 1, 2)  # -1表示自动计算该维度大小

cv2.destroyAllWindows()
capture.release()