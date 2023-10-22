import cv2
import numpy as np
import random 

def generate_random_threeD_points(n):
    threeDpoints = np.zeros((1, n, 4), np.float32)
    for i in range(n):
        threeDpoints[0,i,:] = np.array([random.randint(0,5), random.randint(0,5), random.randint(7,8), 1])

    threeDpoints = np.random.normal(size = (n,3))
    threeDpoints[:,2:] = threeDpoints[:,2:] + 7*np.ones((n,1))
    threeDpoints = np.concatenate((threeDpoints, np.ones((n,1))), axis=1)[np.newaxis,:]
    
    return threeDpoints

def getImagePointsFromthreeD(threeDpoints, P, P_prime):
    
    n = threeDpoints.shape[1]
    
    imagePoints = np.zeros((1, n, 3), np.float32)
    imagePoints_prime = np.zeros((1, n, 3), np.float32)

    for i in range(n):
        imagePoints[0,i,:] = P@threeDpoints[0,i,:]
        imagePoints_prime[0,i,:] = P_prime@threeDpoints[0,i,:]
    #print(imagePoints)
    #print(imagePoints_prime)
    
    return imagePoints, imagePoints_prime
    
def calcE_ransac(imagePoints, imagePoints_prime, K):
    imagePoints = imagePoints.squeeze(0)[:,:2]
    imagePoints_prime = imagePoints_prime.squeeze(0)[:,:2]
    #print(imagePoints)
    #print(imagePoints_prime)

    imagePoints = cv2.undistortPoints(np.expand_dims(imagePoints, axis=1), cameraMatrix=K, distCoeffs=None)
    imagePoints_prime = cv2.undistortPoints(np.expand_dims(imagePoints_prime, axis=1), cameraMatrix=K, distCoeffs=None)

    [ransac_E, mask] = cv2.findEssentialMat(imagePoints, imagePoints_prime);
    
    return ransac_E

def calcE_SVD(imagePoints, imagePoints_prime, K, K_prime):
    A = np.zeros((8,9), dtype=np.float32)

    for i in range(8):
        imagePoints[0,i,:] = imagePoints[0,i,:]/imagePoints[0,i,2]
        imagePoints_prime[0,i,:] = imagePoints_prime[0,i,:]/imagePoints_prime[0,i,2]
        #print(imagePoints[0,i,:])
        #print(imagePoints_prime[0,i,:])
        x = imagePoints[0,i,0]
        y = imagePoints[0,i,1]
        xp = imagePoints_prime[0,i,0]
        yp = imagePoints_prime[0,i,1]
        z = imagePoints[0,i,2]
        zp = imagePoints_prime[0,i,2]
        if z==1 and zp==1:
            A[i,0] = xp*x
            A[i,1] = xp*y
            A[i,2] = xp
            A[i,3] = yp*x
            A[i,4] = yp*y
            A[i,5] = yp
            A[i,6] = x
            A[i,7] = y
            A[i,8] = 1
        else:
            print("Z Error found in point number ", i) 
            

    u, w, vt = np.linalg.svd(A)  #cv2.SVDecomp(A)
    #print("A:", A)
    #print("vt: ", vt.shape)
    #print("w:", w)
    F = vt[8,:]/vt[8,8]
    #print("A*F:", A@F.T)
    F = F.reshape(3,3)
    F = F/F[1,0]
    print("F:", F)
    #for i in range(8):
    #    print(imagePoints_prime[0,i,:].T @ F @ imagePoints[0,i,:])

    # Use findEssentialMat() to get E matrix. It uses 5 point alogorithm.
    E = K_prime.T @ F @ K
    
    return E

def getReconstructedP_Matrix(E):
    U, W, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]])
    u3 = U[:,2][:,np.newaxis]
    #print("u3:\n", u3.shape)
    P_prime_1 = np.concatenate((U@W@Vt, u3), axis=1)
    P_prime_2 = np.concatenate((U@W@Vt, -u3), axis=1)
    P_prime_3 = np.concatenate((U@W.T@Vt, u3), axis=1)
    P_prime_4 = np.concatenate((U@W.T@Vt, -u3), axis=1)

    if P_prime_1[0,0]<0:
        P_prime_1 = -P_prime_1
    if P_prime_2[0,0]<0:
        P_prime_2 = -P_prime_2
    if P_prime_3[0,0]<0:
        P_prime_3 = -P_prime_3
    if P_prime_4[0,0]<0:
        P_prime_4 = -P_prime_4
     
    return P_prime_1, P_prime_2, P_prime_3, P_prime_4

def getNormalizedP_Matrix(P_prime, K_prime):
    P_prime_normalized = np.linalg.inv(K_prime) @ P_prime
    P_prime_normalized_unit_translation = P_prime_normalized.copy()
    P_prime_normalized_unit_translation[:,3] = P_prime_normalized[:,3]/np.linalg.norm(P_prime_normalized[:,3])
    
    return P_prime_normalized_unit_translation

def null(P):
    U,W,Vt = np.linalg.svd(P)
    h = Vt[-1,:]/Vt[-1,-1]
    return h

def getReconstructed_X(point_x, point_x_prime, P, P_prime):
    x = point_x[0]
    y = point_x[1]
    xp = point_x_prime[0]
    yp = point_x_prime[1]

    A = np.zeros((4, P.shape[1]))
    A[0,:] = y*P[2,:] - P[1,:]
    A[1,:] = P[0,:] - x*P[2,:]
    A[2,:] = yp*P_prime[2,:] - P_prime[1,:]
    A[3,:] = P_prime[0,:] - xp*P_prime[2,:]

    X = null(A)
    X = X/X[3]
    return X

def triangulate(P, P_prime, imagePoints, imagePoints_prime):
    
    reconstructed_X = []
    for i in range(imagePoints.shape[1]):
        point_x = imagePoints[0,i,:]
        point_x_prime = imagePoints_prime[0,i,:]
        X = getReconstructed_X( point_x, point_x_prime, P, P_prime )
        reconstructed_X.append( X )
        
    reconstructed_X = np.array(reconstructed_X)
        
    return reconstructed_X

def getNumberCorrectPoints(reconstructed_X, P, P_prime):
    correct = 0
    for X in reconstructed_X:
        w = (P @ X)[2]
        M = P[:,:3]
        M_prime = P_prime[:,:3]
        
        d1 = np.sign(np.linalg.det(M)) * w / X[3]
        d2 = np.sign(np.linalg.det(M_prime)) * w / X[3]
        if d1>0 and d2>0:
            correct+=1
    return correct


def getCorrectP_prime(P_options, imagePoints, imagePoints_prime):
    
    P = np.concatenate((np.eye(3), np.zeros((3,1))), axis=1)
    n_correct = []
    for P_prime in P_options:
        reconstructed_X = triangulate(P, P_prime, imagePoints, imagePoints_prime)
        #print("\nreconstructed_X")
        #print(reconstructed_X)
        correct = getNumberCorrectPoints( reconstructed_X, P, P_prime )
        n_correct.append(correct)
        
    n = np.argmax(n_correct)
    return P_options[n]
    
def DLT_P(imagePoints_prime, threeDpoints):
    
    n = imagePoints_prime.shape[1]
    M = np.zeros((2*n,12))
    print("n: ",n)
    for i in range(n):
        x_i = imagePoints_prime[0,i,0]
        y_i = imagePoints_prime[0,i,1]
        X = threeDpoints[i,:]
        M[2*i, :4] = -X
        M[2*i, 4:8] = np.array([0,0,0,0])
        M[2*i, 8:] = x_i*X
        
        M[2*i+1, :4] = np.array([0,0,0,0])
        M[2*i+1, 4:8] = -X
        M[2*i+1, 8:] = y_i*X
    
    P = null(M).reshape(3,4)
    return P
        
def main():
    
    n = 8               # No. of points
    threeDpoints = generate_random_threeD_points(n)
    
    K = np.array([[960,0,960],[0,960,540],[0,0,1]])
    K_prime = K

    R = np.eye(3)
    t = np.zeros((3,1))
    t_prime = np.array([[-1],[0],[-3]])

    P = K @ np.concatenate((R,t), axis=1)
    P_prime = K_prime @ np.concatenate((R,t_prime), axis=1)

    print("P:\n", P)
    print("P_prime:\n", P_prime)
    
    imagePoints, imagePoints_prime = getImagePointsFromthreeD(threeDpoints, P, P_prime)
    

    E = calcE_SVD(imagePoints, imagePoints_prime, K, K_prime)
    print("Essential Matrix from K and F:")
    print(E)

    ransac_E = calcE_ransac(imagePoints, imagePoints_prime, K)
    print("Essential Matrix from RANSAC 5-point algorithm")
    print(ransac_E)

    P_prime_1, P_prime_2, P_prime_3, P_prime_4 = getReconstructedP_Matrix(E)
    
    P_prime_normalized = getNormalizedP_Matrix(P_prime, K_prime)
    P_prime_1_normalized = getNormalizedP_Matrix(P_prime_1, K_prime)
    
    print('P_prime_normalized')
    print(P_prime_normalized)
    #print('P_prime_1_normalized')
    #print(P_prime_1_normalized)
    
    P_options = (P_prime_1, P_prime_2, P_prime_3, P_prime_4)
    reconstructed_P_prime = getCorrectP_prime(P_options, imagePoints, imagePoints_prime)
    print("Correct P_prime")
    print(reconstructed_P_prime)

    reconstructed_X=[]
    for i in range(imagePoints_prime.shape[1]):
        reconstructed_X.append( np.linalg.pinv(reconstructed_P_prime) @ imagePoints_prime[0,i,:] )
    reconstructed_X = np.array(reconstructed_X)
    reconstructed_X = reconstructed_X/reconstructed_X[:,-1].reshape(-1,1)
    print(threeDpoints)
    print(reconstructed_X)
    
    P3 = DLT_P(imagePoints_prime, reconstructed_X)
    print(P3)
    
    example_threeD_points = generate_random_threeD_points(n)
    print(example_threeD_points.shape)
    print(reconstructed_P_prime.shape)
    print(P3.shape)
    imagePoints, imagePoints_prime = getImagePointsFromthreeD(example_threeD_points, reconstructed_P_prime, P3)
    reconstructed_X = triangulate(reconstructed_P_prime, P3, imagePoints, imagePoints_prime)
    print(example_threeD_points)
    print(reconstructed_X)
    C = np.array([3,0.5,3,1])[:,np.newaxis]
    print(P3@C)
    
    return

main()

