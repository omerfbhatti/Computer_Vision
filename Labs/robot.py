
import math
import numpy as np
import cv2

# Robot camera position

cam_ctr = np.array([[0], [0.4], [0.5]])
cam_rotmat, _ = cv2.Rodrigues(np.array([1, 0, 0]) * 120/180 * math.pi)
Trc = np.eye(4)
Trc[0:3, 0:3] = cam_rotmat
Trc[0:3, 3:4] = - cam_rotmat @ cam_ctr
print("Trc")
print(Trc)
# Test robot origin in camera coordinate system
# print(Trc @ np.array([[0], [0], [0], [1]]))

# Initial robot position: no rotation, origin is at (1, 2, 0) in world frame

prev_pos = np.array([[1], [2], [0]])
Twr1 = np.eye(4)
Twr1[0:3, 3:4] = - prev_pos
print("Twr1")
print(Twr1)
# Initial camera position

Twc1 = Trc @ Twr1
print('s_t-1:', Twc1)

# Test a point in world frame converted to camera frame

# x1 = np.array([[1], [3], [0.5], [1]])
# print(Trc @ np.array([[0],[1],[0.5],[1]]))
# print(Twc1 @ x1)

# Forward motion of 20cm is a translation of points by -0.2 in y

control_vec = np.array([[0], [0.2], [0], [0], [0], [0]])

def log(vec):
  rot_mat, _ = cv2.Rodrigues(vec[3:])
  mat = np.eye(4)
  mat[0:3, 0:3] = rot_mat
  mat[0:3, 3:4] = - rot_mat @ vec[0:3]
  return mat

control_mat = log(control_vec)
print("control_mat")
print(control_mat)

Twc2 = Trc @ log(control_vec) @ np.linalg.inv(Trc) @ Twc1
print('s_t:', Twc2)
