import cv2
import numpy as np

print(cv2.__version__)
size = 300, 600, 3
image = np.zeros(size, dtype=np.uint8)
cv2.circle(image, (250,150), 100, (0,255,128), -100)
cv2.circle(image, (350,150), 100, (255,255,255), -100)
cv2.imshow("", image)
cv2.waitKey(0)