import numpy as np
import time
from stats import Stats
from utils import *
from homography import Homography

akaze_thresh:float = 3e-6 # AKAZE detection threshold set to locate about 1000 keypoints
ransac_thresh:float = 2.5 # RANSAC inlier threshold
nn_match_ratio:float = 0.8 # Nearest-neighbour matching ratio
bb_min_inliers:int = 10 # Minimal number of inliers to draw bounding box
stats_update_period:int = 10 # On-screen statistics are updated every 10 frames

class Tracker:
    def __init__(self, detector, matcher):
        self.detector = detector
        self.matcher = matcher
        self.first_frame=None
        self.camParams = cameraParameters("camera_parameters.yml")
        self.new_bb = None
        #self.homographyData = Homography("robot-homography.yml")
        
    def setFirstFrame(self, frame, boxes, title:str):
        stat = Stats()
        self.first_frame = frame.copy()
        ptContain, obj_masks = self.get_bbox_points_masks(boxes)
        
        # compute the descriptors with AKAZE
        first_kp, first_desc = self.detector.detectAndCompute(self.first_frame, None)
        self.obj_kps, self.obj_desc = self.get_objectKeypoints(obj_masks, first_kp, first_desc)
        
        res = frame.copy()
        for kp, ptC in zip(self.obj_kps, ptContain):
            res = cv2.drawKeypoints(res, kp, None, color=(255,0,0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            drawBoundingBox(res, ptC);
        
        #cv2.imshow("kps", res)
        #cv2.waitKey(0)
        #cv2.destroyWindow("kps")
        
        stat.keypoints = len(first_kp)
        
        cv2.putText(self.first_frame, title, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        self.object_bbs = ptContain  # Object bounding boxes
        return stat
        
    def get_bbox_points_masks(self, boxes):
        iSize = 4
        ptContain = np.zeros((len(boxes), iSize, 2))
        obj_masks = np.zeros((len(boxes),self.first_frame.shape[0], self.first_frame.shape[1], 3), dtype=np.uint8)
        for i, bb in enumerate(boxes):
            ptContain[i,0, :] = np.array([ bb[0]-10, bb[1]-10 ], dtype=np.int32)
            ptContain[i,1, :] = np.array([ bb[0]+bb[2]+10, bb[1]-10 ], dtype=np.int32)
            ptContain[i,2, :] = np.array([ bb[0]+bb[2]+10, bb[1]+bb[3]+10 ], dtype=np.int32)
            ptContain[i,3, :] = np.array([ bb[0]-10, bb[1]+bb[3]+10 ], dtype=np.int32)
            cv2.fillPoly(obj_masks[i], np.int32([ptContain[i]]), (255,255,255))
        
        return ptContain, obj_masks

    def get_objectKeypoints(self, obj_masks, kps, descriptors):
        obj_kps, obj_desc = [], []
        for obj_mask in obj_masks:
            # SELECT POINTS IN THE BOUNDING BOX
            first_kp, first_desc = [], []
            for k, d in zip(kps, descriptors):
                x, y = k.pt  
                if obj_mask[int(y), int(x), 0] != 0:
                    first_kp.append(k)   # Append keypoint to a list of "good keypoint".
                    first_desc.append(d)  # Append descriptor to a list of "good descriptors".
            
            first_desc = cv2.UMat(np.array(first_desc))
            obj_kps.append(first_kp)
            obj_desc.append(first_desc)
        
        return obj_kps, obj_desc

    def get_inlierPoints(self, obj_kp, kp, matches):
        matched1 = []
        matched2 = []
        matched1_keypoints = []
        matched2_keypoints = []
        good = []

        for i,(m,n) in enumerate(matches):
            if m.distance < nn_match_ratio * n.distance:    # nearest neighbour matching ratio
                good.append(m)    # append m to good points if 1stPoint_distance<0.8*2ndPoint_distance
                matched1_keypoints.append(obj_kp[matches[i][0].queryIdx]) # matched keypoint from 1st image
                matched2_keypoints.append(kp[matches[i][0].trainIdx]) # matched keypoint from second image

        matched1 = np.float32([ obj_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2) # keypoints from 1st (original) image
        matched2 = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)  # matched keypoints from 2nd image
        
        return good, matched1, matched2, matched1_keypoints, matched2_keypoints

    def sfm_subprocess(self, frame, obj_kp, kp, obj_desc, desc, object_bb):
        stat = Stats()
        start_time = time.time()
        stat.keypoints = len(kp)
        
        matches = self.matcher.knnMatch(obj_desc, desc, k=2)
        good, matched1, matched2, matched1_keypoints, matched2_keypoints = self.get_inlierPoints(obj_kp, kp, matches)
        
        stat.matches = len(matched1)  # Update stats of matched keypoints in image
        
        # Calculate Homography if equat to or more than four points are matched
        homography = None
        if (len(matched1) >= 4):
            homography, inlier_mask = cv2.findHomography(matched1, matched2, cv2.RANSAC, ransac_thresh)
        print("Homgraphy: ", homography)
        print("No. of initially matched keypoints from Tracker: ", len(matched1))
        
        # if matched points are less than 4, then put inliers to zero and return concatenated images
        if (len(matched1) < 4 or homography is None):
            print("first_frame: ", self.first_frame.shape)
            print("frame: ", frame.shape)
            #res = cv2.hconcat([self.first_frame.copy(), frame])
            res = frame
            print("res: ", res.shape)
            stat.inliers = 0
            stat.ratio = 0
            return res, stat
        
        inlier_matches, inliers1, inliers2, inliers1_keypoints, inliers2_keypoints = \
                                                                                    get_inliers(inlier_mask, good, matched1, matched2, matched1_keypoints, matched2_keypoints)
        
        print("No. of inlier keypoints passing the distance test: ", len(inliers1))
        '''
        if len(inliers1)>=8:
            E, E_mask, F = self.calcE_ransac(self.camParams.cameraMatrix, self.camParams.dist_coeff, inliers1, inliers2)
            print("E: ", E)

            if E.shape == (3,3):
                inlier_matches, inliers1, inliers2, inliers1_keypoints, inliers2_keypoints = \
                                                                                        self.get_inliers_from_EMatrix(E, E_mask, inliers1, inliers2, inliers1_keypoints, inliers2_keypoints)
                
                print("No. of inliers1 from E matrix calculation: ", inliers1.shape[0])
                
                
                corrected_inliers1, corrected_inliers2 = cv2.correctMatches(E, inliers1.transpose(1,0,2), inliers2.transpose(1,0,2))
                points, R, T, mask = cv2.recoverPose(E, corrected_inliers1, corrected_inliers2, self.camParams.cameraMatrix)
                print("No. of inlier Points:", points)
                
                # Construct P matrix
                P1 = np.concatenate((self.camParams.cameraMatrix @ np.eye(3), self.camParams.cameraMatrix @ np.zeros((3,1))), axis=1)
                P2 = np.concatenate((self.camParams.cameraMatrix@R, self.camParams.cameraMatrix@T), axis=1)
                X3d = cv2.triangulatePoints(P1, P2, corrected_inliers1, corrected_inliers2)
                X3d = X3d[:]/X3d[-1]
                print("Scaled X3d:")
                print(X3d)
           '''  
        stat.inliers = len(inliers1)
        stat.ratio = stat.inliers * 1.0 / stat.matches
        
        bb = np.array([object_bb], dtype=np.float32)
        new_bb = cv2.perspectiveTransform(bb, homography)
        frame_with_bb = frame.copy()
        first_frame = self.first_frame.copy()
        
        if (stat.inliers >= bb_min_inliers):
            drawBoundingBox(frame_with_bb, new_bb[0])
        #res = cv2.drawMatches(first_frame, inliers1_keypoints, frame_with_bb, inliers2_keypoints, 
        #                            inlier_matches, None, matchColor=(255, 0, 0), singlePointColor=(255, 0, 0))
        res = frame_with_bb
        print("first_frame: ", first_frame.shape)
        print("res: ", res.shape)
        dt = time.time() - start_time  # Time passed 
        stat.fps = 1. / dt             # Calculate FPS and update in stats
        
        return res, stat
    
    def process(self, frame):
        stat = Stats()
        kps = self.detector.detect(frame, None)
        #kp = self.undistort(kp)
        kps, desc = self.detector.compute(frame, kps)
        res = frame.copy()
        for (obj_kp, obj_desc, first_bbox) in zip(self.obj_kps, self.obj_desc, self.object_bbs):
            res, stat = self.sfm_subprocess(res, obj_kp, kps, obj_desc, desc, first_bbox)
        
        res = np.concatenate((self.first_frame, res), axis=1)
        #cv2.imshow("res", res)
        #cv2.waitKey(0)
        
        new_bb = []
        
        return res, new_bb, stat

    def getDetector(self):
        return self.detector
 
