
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
matplotlib.use('Agg')

# Assumed actual camera matrices

P = np.array([[960,   0, 960, 0],
              [  0, 960, 540, 0],
              [  0,   0,   1, 0]])

Pprime = np.array([[960,   0, 960, -3840],
                   [  0, 960, 540, -1620],
                   [  0,   0,   1,    -3]])

# Assumed actual calibration matrix for both cameras

K = np.array([[960, 0, 960], [0, 960, 540], [0, 0, 1]])

# 8 random 3D points centered about 8m from first camera and 5m from
# second camera

X = np.random.normal(0.0, 2.0, (3, 8))
X = X + np.array([[0.5], [0], [12.0]])
X = np.concatenate((X, np.ones((1, 8))), 0)

# Projections of the 8 points into cameras 1 and 2

x = P @ X
x = x / np.tile(x[2,:], (3, 1))

xprime = Pprime @ X
xprime = xprime / np.tile(xprime[2,:], (3, 1))

fig = plt.figure(figsize=(12,3))
ax0 = fig.add_subplot(121, title='Image 1')
ax1 = fig.add_subplot(122, title='Image 2')
ax0.set_xlim((0, 1920))
ax0.set_ylim((0, 1080))
ax0.invert_yaxis()
ax1.set_xlim((0, 1920))
ax1.set_ylim((0, 1080))
ax1.invert_yaxis()
ax0.plot(x[0,:], x[1,:], 'ro')
ax1.plot(xprime[0,:], xprime[1,:], 'ro')
fig.savefig('projections.jpg')

# Reconstruct F from the point correspondences

A = np.zeros((8, 9))
for i in range(8):
    this_x = x[:,i]
    this_xprime = xprime[:,i]
    A[i,0] = this_xprime[0] * this_x[0]
    A[i,1] = this_xprime[0] * this_x[1]
    A[i,2] = this_xprime[0]
    A[i,3] = this_xprime[1] * this_x[0]
    A[i,4] = this_xprime[1] * this_x[1]
    A[i,5] = this_xprime[1]
    A[i,6] = this_x[0]
    A[i,7] = this_x[1]
    A[i,8] = 1

(U,D,Vt) = np.linalg.svd(A, full_matrices=True, compute_uv=True)
DD = np.concatenate((np.diag(D), np.zeros((8,1))), 1)
# print('UDVt - A:', U @ DD @ Vt - A)
F = np.reshape(Vt[8,:], (3, 3))
F = F / F[1,0]
print('F:', F)

# Calculate E from F and K

E = K.T @ F @ K

(U,D,Vt) = np.linalg.svd(E, full_matrices=True, compute_uv=True)
print('Singular values (should be 1, 1, 0): %f, %f, %f' %
      (D[0]/D[0], D[1]/D[0], D[2]/D[0]))
print('U:', U)
print('Vt:', Vt)
W = np.zeros((3, 3))
W[0,1] = -1
W[1,0] = 1
W[2,2] = 1
Pprime1 = np.concatenate((U @ W @ Vt, U[:,2].reshape((3, 1))), 1)
Pprime1 /= Pprime1[0,0]
Pprime2 = np.concatenate((U @ W @ Vt, -U[:,2].reshape((3, 1))), 1)
Pprime2 /= Pprime2[0,0]
Pprime3 = np.concatenate((U @ W.T @ Vt, U[:,2].reshape((3, 1))), 1)
Pprime3 /= Pprime3[0,0]
Pprime4 = np.concatenate((U @ W.T @ Vt, -U[:,2].reshape((3, 1))), 1)
Pprime4 /= Pprime4[0,0]

print('If P = [I | 0], Pprime is one of the following:')
print(Pprime1)
print(Pprime2)
print(Pprime3)
print(Pprime4)

print('Actual Pprime was:', np.linalg.inv(K) @ Pprime)

