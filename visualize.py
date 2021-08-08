import cv2

image = cv2.imread('data/images/image_0005.jpg')

# 48,18,339,146
cv2.rectangle(image, (48, 18), (339, 146), (0,255,0), 2)

cv2.imshow('PREVIEW', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
