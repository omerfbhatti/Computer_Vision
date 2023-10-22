import cv2
import numpy as np
import math

def distance(x, x_prime, F):
    
    l_prime = (F @ x)[:,np.newaxis]
    print(l_prime)
    l_prime = l_prime/l_prime[2,0]

    line = (F.T @ x_prime)[:,np.newaxis]
    print(line)
    line = line/line[2,0]

    distance_x_l = np.abs( np.dot(x,line)/np.sqrt(line[0]**2 + line[1]**2) )
    distance_xp_lp = np.abs( np.dot(x_prime, l_prime)/np.sqrt(l_prime[0]**2 + l_prime[1]**2) )
    
    print("distance_x_l: ", distance_x_l)
    print("distance_xp_lp: ", distance_xp_lp)

def projectPoint(Tcr, point):
    return Tcr@point[:,np.newaxis]

if __name__=="__main__":
    
    x = np.array([1214,618,1]).T
    x_prime = np.array([1638,737,1]).T
    
    F = np.array([[1.722370581754411e-08, -1.062854124971609e-06, 0.0003422508060393362],
                [9.986249798726018e-07, -2.209841487198608e-08, -0.0005831301456256097],
                [-0.0003468896048613532, 0.0008492220586863253, -0.0764901408322496]])

    distance(x, x_prime, F)
    
    rot = np.array([ 1.6390081888720540e+00, 3.1000248925614857e-02,
                    -4.3421312214750228e-02 ])
    trans = np.array([ 8.5053651529399948e-02, 3.8576576882373542e-01,
                        -3.4753891558701172e-01 ])[:, np.newaxis]
    R, _ = cv2.Rodrigues(rot)
    print("R:", R)
    Tcr = np.concatenate((R, trans), axis=1)
    Tcr = np.concatenate((Tcr, np.array([0,0,0,1])[np.newaxis,:]), axis=0)
    print("Tcr: ", Tcr)
    
    point = np.array([0,0,0,1])
    projected_point = projectPoint(Tcr, point)
    print("Projected Point: ", projected_point)

    
