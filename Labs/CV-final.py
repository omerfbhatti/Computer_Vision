import cv2
import numpy as np

def null(A):
    U,W,Vt = np.linalg.svd(A)
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
    
if __name__=="__main__":
    K = np.array([[1000, 0 , 1920*0.5],
                  [0, 1000, 1080*0.5],
                  [0, 0, 1]])
    print("K")
    print(K)
    
    Orc1 = np.array([[-1],
                     [2.5],
                     [0.3]])
    
    Orc2 = np.array([[1],
                     [2.5],
                     [0.3]])
    
    Rc1 = np.array([[1, 0, 0],   # 0 roll and yaw
                    [0, 1, 0],
                    [0, 0, 1]])
    
    Rc2 = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    
    tc1 = -Rc1 @ Orc1
    tc2 = -Rc2 @ Orc2
    
    Tvc1 = np.concatenate((Rc1, tc1), axis=1)
    Tvc2 = np.concatenate((Rc2, tc2), axis=1)
    
    #twv = np.array([[0],
    #                [0],
    #                [0]])
    
    #Rwv = np.eye(3)
    #Twv = np.concatenate((Rwv, twv), axis=1)
    #Twv = np.concatenate((Twv, np.array([[0,0,0,1]])), axis=0)
    
    #Twc1 = Tvc1 @ Twv
    #Twc2 = Tvc2 @ Twv
    
    P1 = K @ Tvc1
    P2 = K @ Tvc2
    
    print("P1")
    print(P1)
    print("P2")
    print(P2)
    
    
    point11 = np.array([[1010+200*0.5],
                       [500+180*0.5],
                       [1]])
    
    point21 = np.array([[980+210*0.5],
                       [510+185*0.5],
                       [1]])
    
    X1 = getReconstructed_X(point11, point21, P1, P2)
    
    print("X")
    print(X1)
    
    point12 = np.array([[1030+210*0.5],
                       [505+190*0.5],
                       [1]])
    
    point22 = np.array([[970+220*0.5],
                       [505+190*0.5],
                       [1]])
    
    X2 = getReconstructed_X(point12, point22, P1, P2)
    
    r_speed = (X2-X1)
    
    
    print("Relative Speed (dx,dy,dz)")
    print(r_speed)

    print("Relative Speed m/sec")
    print(r_speed*30)
    
    host_speed = 10
