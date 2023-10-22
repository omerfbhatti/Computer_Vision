import cv2
import numpy as np

def null(P):
    U,W,Vt = np.linalg.svd(P)
    h = Vt[-1,:]/Vt[-1,-1]
    return h
    
def findEpipole(P_prime, C):
    e = P_prime @ C[:,np.newaxis]
    e = e/e[-1]
    return e

def getCrossMatrix(e):
    crossMatrix = np.array([[0, -e[2,0], e[1,0]],
                         [e[2,0], 0, -e[0,0]],
                         [-e[1,0], e[0,0], 0]])
    return crossMatrix

if __name__=="__main__":
    Pr = np.array([[ 763.037448, 948.626962, -70.027011, -249.063249],
                    [-27.153472, 413.013589, -834.460406, 146.980511],
                    [-0.047150, 0.996552, -0.068276, -0.347539 ]] )
    
    Pr_prime = np.array([[ 529.243367, 1096.649189, -65.460341, -541.617311],
                        [-114.715323, 397.195667, -834.696173, 54.425526],
                        [-0.270799, 0.960176, -0.068768, -0.625205 ]] )
    
    Tcr = np.array([[0.99887, 0.04661, -0.00943, 0.08505],
                    [-0.00622, -0.06864, -0.99762, 0.38577],
                    [-0.04715, 0.99655, -0.06828, -0.34754],
                    [0.00000, 0.00000, 0.00000, 1.00000 ]])
    
    P = Pr @ np.linalg.inv(Tcr)
    P_prime = Pr_prime @ np.linalg.inv(Tcr)
    print("P:")
    print(P)
    print("P prime:")
    print(P_prime)
    
    # Finding Camera Matricces
    C = null(P)
    print("C:")
    print(C)
    C_prime = null(P_prime)
    print("C prime:")
    print(C_prime)

    e = findEpipole(P, C_prime)
    e_prime = findEpipole(P_prime, C)
    print("e: ",e)
    print("e prime: ", e_prime)
    
    ep_crossMatrix = getCrossMatrix(e_prime)
    print('ep_crossMatrix')
    print(ep_crossMatrix)
    
    
    R,_ = cv2.Rodrigues(np.array([ 1.6390081888720540e+00, 3.1000248925614857e-02,
                                -4.3421312214750228e-02 ]))
    print("R:", R)
    
    K = P_prime[:3,:3] @ np.linalg.inv(R)
    print("k:", K)
    
    F_old = np.array([[1.722370581754411e-08, -1.062854124971609e-06, 0.0003422508060393362],
                [9.986249798726018e-07, -2.209841487198608e-08, -0.0005831301456256097],
                [-0.0003468896048613532, 0.0008492220586863253, -0.0764901408322496]])
    
    F = ep_crossMatrix @ K @ R @ np.linalg.inv(K)
    print(F)
    print(F_old*13.072)
