
import numpy as np



# Set up ground truth cameras P1, P2



K = np.array([[960, 0, 960], [0, 960, 540], [0, 0, 1]])

R = np.eye(3)

t1 = np.zeros((3, 1))

P1 = K @ np.concatenate([R, t1], 1)

t2 = np.array([[-1], [0], [-3]])

P2 = K @ np.concatenate((R, t2), 1)



# Get 8 random points around (0, 0, 7) in the world coordinate system



depth = 7

X = np.random.normal(size=(3, 8)) + np.array([[0], [0], [depth]])

X = np.concatenate((X, np.ones((1, 8))), 0)



# Project points into the two images



x1 = P1 @ X

x1 = x1 / np.tile(x1[2:,:], (3, 1))

x2 = P2 @ X

x2 = x2 / np.tile(x2[2:,:], (3, 1))



# Calculate F



A = np.zeros((8, 9))

for i in range(8):

 A[i, 0] = x2[0, i] * x1[0, i]

 A[i, 1] = x2[0, i] * x1[1, i]

 A[i, 2] = x2[0, i] * x1[2, i]

 A[i, 3] = x2[1, i] * x1[0, i]

 A[i, 4] = x2[1, i] * x1[1, i]

 A[i, 5] = x2[1, i] * x1[2, i]

 A[i, 6] = x2[2, i] * x1[0, i]

 A[i, 7] = x2[2, i] * x1[1, i]

 A[i, 8] = x2[2, i] * x1[2, i]

# f is last row of Vt in SVD of A

U, S, Vt = np.linalg.svd(A)

f = Vt[8:,:].T

F = np.reshape(f, (3, 3))



# Check F



# for i in range(8):

# print(x2[:,i:i+1].T @ F @ x1[:,i:i+1])



# E = Kt F K



E = K.T @ F @ K

U, S, Vt = np.linalg.svd(E)

W = np.array([[0.0, -1, 0], [1.0, 0, 0], [0, 0, 1]])



u3 = U[:,2:]

P21 = np.concatenate((U @ W @ Vt, u3), 1)

P22 = np.concatenate((U @ W @ Vt, -u3), 1)

P23 = np.concatenate((U @ W.T @ Vt, u3), 1)

P24 = np.concatenate((U @ W.T @ Vt, -u3), 1)



if P21[0,0] < 0:

 P21 = -P21

if P22[0,0] < 0:

 P22 = -P22

if P23[0,0] < 0:

 P23 = -P23

if P24[0,0] < 0:

 P24 = -P24



P2norm = np.linalg.inv(K) @ P2

print('Orignal P2 normalized:')

print(P2norm)



P2normunit = P2norm.copy()

P2normunit[:,3:] = P2normunit[:,3:] / np.linalg.norm(P2normunit[:,3:])

print('Original P2 normalized with unit translation:')

print(P2normunit)



print('Solution 1:')

print(P21)

print('Solution 2:')

print(P22)

print('Solution 3:')

print(P23)

print('Solution 4:')

print(P24)

