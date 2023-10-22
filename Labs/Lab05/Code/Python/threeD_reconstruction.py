import numpy as np
import time
from stats import Stats
from utils import *
from tracker import Tracker

akaze_thresh:float = 3e-4 # AKAZE detection threshold set to locate about 1000 keypoints
ransac_thresh:float = 2.5 # RANSAC inlier threshold
nn_match_ratio:float = 0.8 # Nearest-neighbour matching ratio
bb_min_inliers:int = 100 # Minimal number of inliers to draw bounding box
stats_update_period:int = 10 # On-screen statistics are updated every 10 frames


def recontruct_points(img1, img2):
    
    
    akaze_stats = Stats()
    orb_stats = Stats()

    akaze = cv2.AKAZE_create()
    akaze.setThreshold(akaze_thresh)

    orb = cv2.ORB_create()

    matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")

    akaze_tracker = Tracker(akaze, matcher)
    orb_tracker = Tracker(orb, matcher)

    cv2.namedWindow("first", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED);
    cv2.imshow("first", img1)
    uBox = cv2.selectROI("first", img1);
    bb = []
    bb.append((uBox[0], uBox[1]))
    bb.append((uBox[0] + uBox[2], uBox[0] ))
    bb.append((uBox[0] + uBox[2], uBox[0] + uBox[3]))
    bb.append((uBox[0], uBox[0] + uBox[3]))
    stat_a = akaze_tracker.setFirstFrame(img1, bb, "AKAZE",);
    stat_o = orb_tracker.setFirstFrame(img1, bb, "ORB");  # returns no. of keypoints detected

    akaze_draw_stats = stat_a.copy()
    orb_draw_stats = stat_o.copy()

    update_stats = True
    
    akaze_res, stat = akaze_tracker.process(img2)
    akaze_stats + stat
    if (update_stats):
        akaze_draw_stats = stat
    orb.setMaxFeatures(stat.keypoints)
    orb_res, stat = orb_tracker.process(img2)
    orb_stats + stat
    if (update_stats):
        orb_draw_stats = stat
    drawStatistics(akaze_res, akaze_draw_stats)
    drawStatistics(orb_res, orb_draw_stats)
    res_frame = cv2.vconcat([akaze_res, orb_res])
    # cv2.imshow(video_name, akaze_res)
    cv2.imshow("first", res_frame)
    cv2.waitKey(0)

    akaze_stats / (1)
    orb_stats / (1)
    printStatistics("AKAZE", akaze_stats);
    printStatistics("ORB", orb_stats);
    return 0

if __name__=="__main__":
    img1 = cv2.imread("frame-013.png")
    img2 = cv2.imread("frame-017.png")
    recontruct_points(img1, img2)
