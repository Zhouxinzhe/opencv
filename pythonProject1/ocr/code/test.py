import function
import cv2


def cv_show(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# img = cv2.imread('../images/template.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# bounding_box = cv2.boundingRect(contours[0])
# print(bounding_box)

img = cv2.imread('../images/credit_card.jpg')
print(img.shape)
cv_show(img, 'img')
cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 255), 1)
cv2.putText(img, 'Hello World!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
cv_show(img, 'img')
