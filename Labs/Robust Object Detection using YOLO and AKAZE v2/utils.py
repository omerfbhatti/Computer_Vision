from stats import Stats
import cv2
from typing import List #use it for :List[...]
import numpy as np

def drawBoundingBox(image, bb):
    """
    Draw the bounding box from the points set

    Parameters
    ----------
        image (array):
            image which you want to draw
        bb (List):
            points array set
    """
    color = (0, 0, 255)
    for i in range(len(bb) - 1):
        b1 = (int(bb[i][0]), int(bb[i][1]))
        b2 = (int(bb[i + 1][0]), int(bb[i + 1][1]))
        cv2.line(image, b1, b2, color, 2)
    b1 = (int(bb[len(bb) - 1][0]), int(bb[len(bb) - 1][1]))
    b2 = (int(bb[0][0]), int(bb[0][1]))
    cv2.line(image, b1, b2, color, 2)

def drawStatistics(image, stat: Stats):
    """
    Draw the statistic to images

    Parameters
    ----------
        image (array):
            image which you want to draw
        stat (Stats):
            statistic values
    """
    font = cv2.FONT_HERSHEY_PLAIN

    str1, str2, str3, str4, str5 = stat.to_strings()

    shape = image.shape

    cv2.putText(image, str1, (0, shape[0] - 120), font, 2, (0, 0, 255), 3)
    cv2.putText(image, str2, (0, shape[0] - 90), font, 2, (0, 0, 255), 3)
    cv2.putText(image, str3, (0, shape[0] - 60), font, 2, (0, 0, 255), 3)
    cv2.putText(image, str5, (0, shape[0] - 30), font, 2, (0, 0, 255), 3)

def printStatistics(name: str, stat: Stats):
    """
    Print the statistic

    Parameters
    ----------
        name (str):
            image which you want to draw
        stat (Stats):
            statistic values
    """
    print(name)
    print("----------")
    str1, str2, str3, str4, str5 = stat.to_strings()
    print(str1)
    print(str2)
    print(str3)
    print(str4)
    print(str5)
    print()

def Points(keypoints):
    res = []
    for i in keypoints:
        res.append(i)
    return res


def get_inliers_from_EMatrix(self, E, E_mask, inliers1, inliers2, inliers1_keypoints, inliers2_keypoints):
    idx, _ = np.where(E_mask == True)
    new_inlier1_kps = []
    new_inlier2_kps = []
    #print("E_mask:", E_mask.shape)
    #print("idx:", idx.shape)
    #print("oidx:", oidx.shape)
    #print("idx:", idx)
    #print("oidx:", oidx)
    for i in range(len(inliers1_keypoints)):
        if i in idx:
            new_inlier1_kps.append(inliers1_keypoints[i])
            new_inlier2_kps.append(inliers2_keypoints[i])
    
    new_inliers1 = inliers1[idx]
    new_inliers2 = inliers2[idx]
    #print("inliers1: ", new_inliers1.shape)
    #print("inliers2: ", new_inliers2.shape)
    #print("no. of refined inliers: ", new_inliers1.shape)
    
    zeros = []
    K = self.camParams.cameraMatrix
    
    inlier_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0) for idx in range(len(new_inliers1))]
    
    return inlier_matches, new_inliers1, new_inliers2, new_inlier1_kps, new_inlier2_kps

def undistort(self, kp):
    matchpoints = []
    for keypoint in kp:
        k = np.array([keypoint.pt[0], keypoint.pt[1]], dtype=np.float32)
        matchpoints.append(k)
        
    matchpoints = np.array(matchpoints)
    #print("matchpoints: ", matchpoints.shape)
    #print(matchpoints)
    kpoints = cv2.undistortPoints(matchpoints, self.camParams.cameraMatrix, self.camParams.dist_coeff)
    #print("undistorted keypoints: ", kpoints.shape)
    #print("undistorted keypoints: ", kpoints)
    for i, keypoint in enumerate(kp):
        keypoint.pt = (kpoints[i][0][0], kpoints[i][0][1])
    
    return kp
    
def get_inliers(inlier_mask, good, matched1, matched2, matched1_keypoints, matched2_keypoints):
    inliers1 = []
    inliers2 = []
    inliers1_keypoints = []
    inliers2_keypoints = []
    
    # From matched points, extract points classed as inliers through RANSAC 
    for i in range(len(good)):      # Run for every matched point that is significant
        if (inlier_mask[i] > 0):    # inlier_mask computed from RANSAC
            new_i = len(inliers1)
            inliers1.append(matched1[i])
            inliers2.append(matched2[i])
            inliers1_keypoints.append(matched1_keypoints[i])
            inliers2_keypoints.append(matched2_keypoints[i])
            
    inlier_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0) for idx in range(len(inliers1))]
    inliers1 = np.array(inliers1, dtype=np.float32)
    inliers2 = np.array(inliers2, dtype=np.float32)
    
    return inlier_matches, inliers1, inliers2, inliers1_keypoints, inliers2_keypoints

def calcE_ransac(K, dist_coeff, imagePoints, imagePoints_prime):
    
    #imagePoints = cv2.undistortPoints(imagePoints, cameraMatrix=K, distCoeffs=dist_coeff)
    #imagePoints_prime = cv2.undistortPoints(imagePoints_prime, cameraMatrix=K, distCoeffs=dist_coeff)
    
    F, mask_F = cv2.findFundamentalMat(imagePoints, imagePoints_prime, cv2.FM_RANSAC)
    ransac_E, mask = cv2.findEssentialMat(imagePoints, imagePoints_prime, K)
    
    return ransac_E, mask, F


class cameraParameters:
    def __init__(self, camera_parameters_file):
        self.cameraMatrix = []
        self.h = 0
        self.w = 0
        self.dist_coeff = []
        self.rvecs = []
        self.tvecs = []
        self.read(camera_parameters_file)
        
    def read(self, camera_parameters_file):
        fileStorage = cv2.FileStorage(camera_parameters_file, cv2.FILE_STORAGE_READ)
        if not fileStorage.isOpened():
            return False

        self.cameraMatrix = fileStorage.getNode("Camera_matrix").mat()
        self.dist_coeff = fileStorage.getNode("dist").mat()
        self.h = int(fileStorage.getNode("image_height").real())
        self.w = int(fileStorage.getNode("image_width").real())
        self.rvecs = fileStorage.getNode("rvecs").mat().squeeze(0)
        self.tvecs = fileStorage.getNode("tvecs").mat().squeeze(0)
        fileStorage.release()
        return True
        
def openCalibrationSettings(filename):

    try:
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        h = fs.getNode("image_height")
        w = fs.getNode("image_width")
        mtx = fs.getNode("Camera_matrix").mat()
        dist = fs.getNode("dist").mat()
        rvecs = fs.getNode("rvecs").mat().squeeze(0)
        tvecs = fs.getNode("tvecs").mat().squeeze(0)
        print("Camera matrix : \n")
        print(mtx)
        print("dist : \n")
        print(dist)
        # print("rvecs : \n")
        # print(rvecs)
        # print("tvecs : \n")
        # print(tvecs)
        fs.release()
        return h, w, mtx, dist, rvecs, tvecs
    except:
        print("Error occured in reading file.")
        raise ValueError()

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
        
    
    P_options = (P_prime_1, P_prime_2, P_prime_3, P_prime_4)
    
    return P_options
        
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

def showUndistorted(img, mtx, dist):
    # Return undistorted images
    
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    res = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    
    # crop the image
    x, y, w, h = roi
    res = res[y:y+h, x:x+w]
    return res

        
