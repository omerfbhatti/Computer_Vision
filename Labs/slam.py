import cv2
import numpy as np

def calc_log(u):
    ''' Expand dimensions to the manifold '''
    
    rotMat = u[3:,:]
    rotMat,_ = cv2.Rodrigues(rotMat)  # Get Rotation Matrix
    t_vector = -rotMat @ u[:3,:]      # Get World Origin in Camera Coordinate System
    T = np.concatenate((rotMat, t_vector), axis=1)
    T = np.concatenate((T, np.array([[0,0,0,1]])), axis=0)
    return T
    

def main():
    # Current Positions
    o_rw = np.array([[1],[2],[0]])      # Location of Robot in World Coordinates
    o_cr = np.array([[0],[0.4],[0.5]])  # Location of Camera in Robot Coordinates
    
    # Angle of rotation for the axis frame
    # 30 degrees tilt downwards about the x-axis
    alpha = np.radians(120)   # 120 degrees about the x-axis
    
    
    # ROTATIONS
    R_wr = np.eye(3)     # Rotation World Coordinates to Robot Coordinates
    rtwist = np.array([[1],[0],[0]]) * alpha 
    R_rc, _ = cv2.Rodrigues(rtwist)    # Rotation Robot Coordinates to Camera Coordinates
    
    # Transformations
    T_wr = np.concatenate((R_wr, -R_wr@o_rw), axis=1)
    T_wr = np.concatenate((T_wr, np.array([[0,0,0,1]])), axis=0)
    T_rc = np.concatenate((R_rc, -R_rc@o_cr), axis=1)
    T_rc = np.concatenate((T_rc, np.array([[0,0,0,1]])), axis=0)
    #print("T_wr")
    #print(T_wr)
    #print("T_rc")
    #print(T_rc)
    T_wc_0 = T_rc @ T_wr  # First Transform from world to robot; then from robot to camera
    
    # State at time t = Transformation matrix at time t
    state_t = T_wc_0
    print("Initial state")
    print(state_t)
    
    # Control Vector at time t
    # First three elements denote 3D position
    # Last three elements denote rotation
    u = np.array([[0],[0.2],[0],[0],[0],[0]])
    
    # Generate Random movement vectors
    controls = np.random.rand(5,6,1)
    
    for u in controls:
        
        # Use control vector and do expansion into manifold dimensions 
        # to get Control Matrix
        log_u = calc_log(u)
        
        print("u")
        print(u)
        # print("log(u)")
        # print(log_u)
        
        # State at time t+1
        # Control Matrix log_u multiplied with previous T_wr to get updated T_wr
        # As T_wr = inv(T_rc) @ T_wc and state_t = T_wc_0
        # Then updated T_wr multiplied with T_rc to get updated T_wc
        # Updated T_wc --> updated state
        # This formulation is used because the T_rc remains constant
        # and the state (T_wc because of change in T_wr) keeps changing. 
        # So no need to recalculate T_wr separately each time
        state_tplus1 = T_rc @ log_u @ np.linalg.inv(T_rc) @ state_t
        print("state(t+1): ")
        print(state_tplus1)
        state_t = state_tplus1
    
main()
