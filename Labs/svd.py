# Python SVD example

import numpy as np
import cv2

matA = np.array([[3.0,2.0,4.0],[8.0,4.0,2.0],[1.0,3.0,2.0]])
w, u, vt = cv2.SVDecomp(matA)
print("A:")
print(matA)
print("U:")
print(u)
print("W:")
print(w)
print("Vt:")
print(vt)

matA = u@w@vt.T
