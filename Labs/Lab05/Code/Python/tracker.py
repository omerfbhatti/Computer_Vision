import numpy as np
import time
from stats import Stats
from utils import *
from homography import Homography

akaze_thresh:float = 3e-4 # AKAZE detection threshold set to locate about 1000 keypoints
ransac_thresh:float = 2.5 # RANSAC inlier threshold
nn_match_ratio:float = 0.8 # Nearest-neighbour matching ratio
bb_min_inliers:int = 100 # Minimal number of inliers to draw bounding box
stats_update_period:int = 10 # On-screen statistics are updated every 10 frames

class Tracker:
    def __init__(self, detector, matcher):
        self.detector = detector
        self.matcher = matcher
        self.camParams = cameraParameters("rpi_camera_parameters.yml")
        self.homographyData = Homography("robot-homography.yml")

    def setFirstFrame(self, frame, bb, title:str):
        iSize = len(bb)
        stat = Stats()
        ptContain = np.zeros((iSize, 2))
        i = 0
        for b in bb:
            #ptMask[i] = (b[0], b[1])
            ptContain[i, 0] = b[0]
            ptContain[i, 1] = b[1]
            i += 1
        
        self.first_frame = frame.copy()
        matMask = np.zeros(frame.shape, dtype=np.uint8)
        cv2.fillPoly(matMask, np.int32([ptContain]), (255,0,0))
        
        #h,  w = frame.shape[:2]
        #self.camParams.cameraMatrix, roi = cv2.getOptimalNewCameraMatrix(self.camParams.cameraMatrix, self.camParams.dist_coeff, (w, h), 1, (w,h))
        
        
        # cannot use in ORB
        # self.first_kp, self.first_desc = self.detector.detectAndCompute(self.first_frame, matMask)
        
        #print("Camera Parameters:")
        #print(self.camParams.cameraMatrix)
        #print("homography data:")
        #print(self.homographyData.matH)
        
        # find the keypoints from the selected image with ORB
        kp = self.detector.detect(self.first_frame,None)
        
        #kp = self.undistort(kp)
        
        # compute the descriptors with ORB
        self.first_kp, self.first_desc = self.detector.compute(self.first_frame, kp)
        
        
        
        print("Print total keypoints: ", len(self.first_kp))
        #print("Printing first detected Keypoint from first Frame:")
        #print("x: ", self.first_kp[0].pt[0])
        #print("y: ", self.first_kp[0].pt[1])
        #print("angle of rotation: ", self.first_kp[0].angle)
        #print("size: ", self.first_kp[0].size)
        
        res = cv2.drawKeypoints(self.first_frame, self.first_kp, None, color=(255,0,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        stat.keypoints = len(self.first_kp)
        #drawBoundingBox(self.first_frame, bb);

        cv2.imshow("key points of {0}".format(title), res)
        cv2.waitKey(0)
        cv2.destroyWindow("key points of {0}".format(title))

        cv2.putText(self.first_frame, title, (0, 60), cv2.FONT_HERSHEY_PLAIN, 5, (0,0,0), 4)
        self.object_bb = bb  # Object bounding box
        return stat

        
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
        
    def get_inlierPoints(self, kp, matches):
        matched1 = []
        matched2 = []
        matched1_keypoints = []
        matched2_keypoints = []
        good = []

        for i,(m,n) in enumerate(matches):
            if m.distance < nn_match_ratio * n.distance:    # nearest neighbour matching ratio
                good.append(m)    # append m to good points if 1stPoint_distance<0.8*2ndPoint_distance
                matched1_keypoints.append(self.first_kp[matches[i][0].queryIdx]) # matched keypoint from 1st image
                matched2_keypoints.append(kp[matches[i][0].trainIdx]) # matched keypoint from second image

        matched1 = np.float32([ self.first_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2) # keypoints from 1st (original) image
        matched2 = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)  # matched keypoints from 2nd image
        
        return good, matched1, matched2, matched1_keypoints, matched2_keypoints
        
    def get_inliers(self, inlier_mask, good, matched1, matched2, matched1_keypoints, matched2_keypoints):
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
    
    def calcE_ransac(self, imagePoints, imagePoints_prime):
        K = self.camParams.cameraMatrix
        dist_coeff = self.camParams.dist_coeff

        #imagePoints = cv2.undistortPoints(imagePoints, cameraMatrix=K, distCoeffs=dist_coeff)
        #imagePoints_prime = cv2.undistortPoints(imagePoints_prime, cameraMatrix=K, distCoeffs=dist_coeff)
        
        F, mask_F = cv2.findFundamentalMat(imagePoints, imagePoints_prime, cv2.FM_RANSAC)
        ransac_E, mask = cv2.findEssentialMat(imagePoints, imagePoints_prime)
        
        return ransac_E, mask, F

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
        
        for i in range(len(new_inlier1_kps)):
            X = np.array([new_inliers1[i][0,0], new_inliers1[i][0,1], 1])[:,np.newaxis]
            Xp = np.array([new_inliers2[i][0,0], new_inliers2[i][0,1], 1])[:,np.newaxis]
            #print("X: ", X.shape)
            val = X.T@np.linalg.inv(K.T)@E@np.linalg.inv(K)@Xp
            zeros.append(val.item())
            
        print("\nX.T@np.linalg.inv(K.T)@E@np.linalg.inv(K)@X':")
        print(zeros)
        
        inlier_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0) for idx in range(len(new_inliers1))]
        
        return inlier_matches, new_inliers1, new_inliers2, new_inlier1_kps, new_inlier2_kps
        
    def drawLine(self, lines1, lines2, img):
        _, c, _ = img.shape
        line1 = lines1[0]
        line2 = lines2[0]
        
        K = self.camParams.cameraMatrix
        dist_coeff = self.camParams.dist_coeff
        
        ya = 0
        x0,y0 = map(int, [0, -line1[2]/line1[1] + ya ])
        x1,y1 = map(int, [c,  ya+(-(line1[2]+line1[0]*c)/line1[1]) ])
        x = np.array([[x0,y0],[x1,y1]])
        x = x[:,np.newaxis,:].astype(np.float32)
        #print("x shape: ", x.shape)
        
        #x = cv2.undistortPoints(x, cameraMatrix=K, distCoeffs=dist_coeff)
        x = x.astype(np.int32)
        #print(x)
        #img2 = cv2.line(img, (x0,y0), (x1,y1), (0, 0, 255), 5)
        img2 = cv2.line(img, (x[0,0,0],x[0,0,1]), (x[1,0,0],x[1,0,1]), (0, 0, 255), 5)
        
        x0,y0 = map(int, [0, -line2[2]/line2[1] + ya ])
        x1,y1 = map(int, [c, ya + (-(line2[2]+line2[0]*c)/line2[1]) ])
        x = np.array([[x0,y0],[x1,y1]])
        x = x[:,np.newaxis,:].astype(np.float32)
        #print("x shape: ", x.shape)
        #x = cv2.undistortPoints(x, cameraMatrix=K, distCoeffs=dist_coeff)
        x = x.astype(np.int32)
        #print(x)
        #img2 = cv2.line(img, (x0,y0), (x1,y1), (0, 0, 255), 5)
        img2 = cv2.line(img2, (x[0,0,0],x[0,0,1]), (x[1,0,0],x[1,0,1]), (0, 0, 255), 5)
        #img2 = cv2.line(img2, (x0,y0), (x1,y1), (0, 0, 255), 5)
        
        return img2


    def process(self, frame):
        stat = Stats()
        start_time = time.time()
        
        kp = self.detector.detect(frame, None)
        #kp = self.undistort(kp)
        kp, desc = self.detector.compute(frame, kp)
        
        #kp, desc = self.detector.detectAndCompute(frame, None)
        stat.keypoints = len(kp)
        matches = self.matcher.knnMatch(self.first_desc, desc, k=2)
        #print("matches:", matches[0])
        
        good, matched1, matched2, matched1_keypoints, matched2_keypoints = self.get_inlierPoints(kp, matches)
        
        stat.matches = len(matched1)  # Update stats of matched keypoints in image
        
        # Calculate Homography if equat to or more than four points are matched
        homography = None
        if (len(matched1) >= 4):
            homography, inlier_mask = cv2.findHomography(matched1, matched2, cv2.RANSAC, ransac_thresh)
        print("Homgraphy: ", homography)
        print("No. of initially matched keypoints from Tracker: ", len(matched1))
        dt = time.time() - start_time  # Time passed 
        stat.fps = 1. / dt             # Calculate FPS and update in stats
        
        # if matched points are less than 4, then put inliers to zero and return concatenated images
        if (len(matched1) < 4 or homography is None):
            res = cv2.hconcat([self.first_frame, frame])
            stat.inliers = 0
            stat.ratio = 0
            return res, stat
        
        inlier_matches, inliers1, inliers2, inliers1_keypoints, inliers2_keypoints = self.get_inliers(inlier_mask, good,                                        
                                                                                                            matched1, matched2, matched1_keypoints, matched2_keypoints)
        print("No. of inlier keypoints passing the distance test: ", len(inliers1))
        #E, E_mask = cv2.findEssentialMat(inliers1, inliers2) #, self.camParams.cameraMatrix, 'RANSAC')
        #print("E: ", E)
        
        E, E_mask, F = self.calcE_ransac(inliers1, inliers2)
        
        
        print("F: ", F)
        print("E: ", E)
        # print("total correct in E:", mask.shape, mask.sum())
        inlier_matches, inliers1, inliers2, inliers1_keypoints, inliers2_keypoints = self.get_inliers_from_EMatrix(E, E_mask, inliers1, inliers2, inliers1_keypoints, inliers2_keypoints)
        
        print("No. of inliers1 from E matrix calculation: ", inliers1.shape[0])
        
        P_options = getReconstructedP_Matrix(E)
        P = getCorrectP_prime(P_options, inliers1, inliers2)
        #print("P: ", P)
        R = P[:,:3]
        T = P[:,3:]
        print("R_from Essential Matrix Decomposition: ", R)
        print("T_from Essential Matrix Decomposition: ", T)
        
        corrected_inliers1, corrected_inliers2 = cv2.correctMatches(E, inliers1.transpose(1,0,2), inliers2.transpose(1,0,2))
        points, R, T, mask = cv2.recoverPose(E, corrected_inliers1, corrected_inliers2, self.camParams.cameraMatrix)
        print("R from recoverPose: ", R)
        print("T from recoverPose: ", T)
        print("No. of inlier Points:", points)
        
        # Construct P matrix
        P1 = np.concatenate((self.camParams.cameraMatrix @ np.eye(3), self.camParams.cameraMatrix @ np.zeros((3,1))), axis=1)
        P2 = np.concatenate((self.camParams.cameraMatrix@R, self.camParams.cameraMatrix@T), axis=1)
        print("C: ", null(P2))
        X3d = cv2.triangulatePoints(P1, P2, corrected_inliers1, corrected_inliers2)
        print("X3d:")
        print(X3d)
        
        # Scale ambiguity
        X3d_cor = X3d[:]/X3d[-1]
        z = X3d_cor[2,:]
        groundPixel_z = np.min(z)
        cameraHeight = T[2]-np.min(z)
        T[2] = cameraHeight
        print("Z: ", np.min(z))
        print("camera Height: ", cameraHeight)
        # Reconstruct P matrix
        T1 = np.array([[0],[0],[cameraHeight+np.min(z)]], dtype=np.float32)
        P1 = np.concatenate((self.camParams.cameraMatrix @ np.eye(3), self.camParams.cameraMatrix @ T1), axis=1)
        P2 = np.concatenate((self.camParams.cameraMatrix@R, self.camParams.cameraMatrix@T), axis=1)
        
        X3d = cv2.triangulatePoints(P1, P2, corrected_inliers1, corrected_inliers2)
        print("Scaled X3d:")
        print(X3d)
        
        stat.inliers = len(inliers1)
        stat.ratio = stat.inliers * 1.0 / stat.matches
        
        bb = np.array([self.object_bb], dtype=np.float32)
        new_bb = cv2.perspectiveTransform(bb, homography)
        frame_with_bb = frame.copy()
        first_frame = self.first_frame.copy()
        
        #frame_with_bb = showUndistorted(frame_with_bb, self.camParams.cameraMatrix, self.camParams.dist_coeff)
        #first_frame = showUndistorted(self.first_frame.copy(), self.camParams.cameraMatrix, self.camParams.dist_coeff)
        if (stat.inliers >= bb_min_inliers):
            drawBoundingBox(frame_with_bb, new_bb[0])
        
        eplines1 = cv2.computeCorrespondEpilines(inliers2.reshape(-1,1,2), 2, F)
        eplines1 = eplines1.reshape(-1, 3)
        
        eplines2 = cv2.computeCorrespondEpilines(inliers1.reshape(-1,1,2), 1, F)
        eplines2 = eplines2.reshape(-1,3)
        
        first_frame = self.drawLine(eplines1, eplines2, first_frame)
        frame_with_bb = self.drawLine(eplines2, eplines1, frame_with_bb)
        
        res = cv2.drawMatches(first_frame, inliers1_keypoints, frame_with_bb, inliers2_keypoints, inlier_matches, None, matchColor=(255, 0, 0), singlePointColor=(255, 0, 0))
        return res, stat

    def getDetector(self):
        return self.detector
 
