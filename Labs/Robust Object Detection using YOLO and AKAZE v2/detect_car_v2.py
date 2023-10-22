import cv2
import numpy as np
import time
from scipy.spatial import distance
from camera import cameraParameters
from homography import Homography
from tracker import Tracker
from stats import Stats
from utils import *

INPUT_FILE = "night_drive/VID-20220723-WA0004.mp4"
OUTPUT_FILE = "night_drive_out_ORB_test_04.mp4"

akaze_thresh:float = 3e-6 # AKAZE detection threshold set to locate about 1000 keypoints
ransac_thresh:float = 2.5 # RANSAC inlier threshold
nn_match_ratio:float = 0.8 # Nearest-neighbour matching ratio
bb_min_inliers:int = 10 # Minimal number of inliers to draw bounding box
stats_update_period:int = 10 # On-screen statistics are updated every 10 frames

class pts:
    def __init__(self, p1_idx, p2_idx, p1, p2, d):
        self.p1_idx = p1_idx
        self.p2_idx = p2_idx
        self.p1 = []
        self.p2 = []
        self.distance = d

class distanceDetector:
    def __init__(self):
        with open('coco.names', 'r') as f:
            self.classes = f.read().splitlines()
        
        net = cv2.dnn.readNetFromDarknet('models/yolov4-tiny.cfg', 'models/yolov4-tiny.weights')

        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
        
        self.akaze_stats = Stats()
        
        detector = cv2.ORB_create()
        #detector.setThreshold(akaze_thresh)
        matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
        self.tracker = Tracker(detector, matcher)
        
        self.akaze_draw_stats = None
        
        
    def detect(self, img):
        start = time.time() 
        classIds, scores, boxes = self.model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
        time_taken = time.time()-start
        print("time taken for inference: ", time_taken)
        
        akaze_img = None
        if self.tracker.first_frame is not None:
            # Returns last object_box calculated from homography found in structure from motion calculations
            akaze_img, self.latest_object_box, running_stats = self.tracker.process(img)
            
        centers = []
        rois = []
        best_distance = np.inf
        new_object_bbox = None
        
        for (classId, score, box) in zip(classIds, scores, boxes):
            print("class ID: ", classId)
            if classId==2:
                if self.tracker.first_frame is not None:
                    d = self.calc_bbox_distance(self.latest_object_box, box)
                    if d < best_distance:
                        best_distance = d
                        new_object_bbox = box
                else:
                    new_object_bbox = box
        
        if new_object_bbox is not None:
            stats_first_frame = self.tracker.setFirstFrame(img, new_object_bbox, "AKAZE Features")
            
        for (classId, score, box) in zip(classIds, scores, boxes):
            #print("class ID: ", classId)
            if classId==2:
                    
                cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                            color=(0, 255, 0), thickness=2)
            
                text = '%s: %.2f' % (self.classes[classId], score)
                cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            color=(0, 255, 0), thickness=2)
                
        
        cv2.putText(img, "YOLO output", (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            color=(0, 255, 0), thickness=2)
        
        if akaze_img is not None:
            img = np.concatenate((img, np.zeros((img.shape), dtype=np.uint8)), axis=1)
            img = np.concatenate((img, akaze_img), axis=0)
        
        return img
    
    def calc_bbox_distance(self, bbox1, bbox2):
        print("bbox1")
        print(bbox1)
        print("bbox2")
        print(bbox2)
        center_1 = ( int(bbox1[0][0] + bbox1[2][0]/2), int(bbox1[0][1] + bbox1[3][1]/2))
        center_2 = ( int(bbox2[0] + bbox2[2]/2), int(bbox2[1] + bbox2[3]/2))
        d = distance.euclidean(center_1, center_2)
        return d
        
    def matchPairwiseDistances(self, centers):
        pairs = []
        for i in range(len(centers)):
            for j in range(i+1,len(centers)):
                pts1 = self.projectPoint(centers[i])
                pts2 = self.projectPoint(centers[j])
                dist_pixel = distance.euclidean(pts1, pts2)
                dist_metric = dist_pixel * 1.333/80
                if dist_metric < 3:
                    print("center1: ", centers[i])
                    print("center2: ", centers[j])
                    print("pts1: ", pts1)
                    print("pts2: ", pts2)
                    print("metric distance: ", dist_metric)
                    print("pixel distance: ", dist_pixel)
                    # print("pair: ", i, ", ", j)
                    pairs.append(pts(i,j, pts1, pts2, dist_metric))
        return pairs
    
    def projectPoint(self, x):
        x = np.array([[x[0]],[x[1]],[1]])
        # print("x")
        # print(x)
        x = self.H @ x
        x = x/x[2,0]
        return x

    
def main():
    img = cv2.imread('demo_data/test.jpg')
    
    detector = distanceDetector()
    img = detector.detect(img)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
def video():
        
    cap = cv2.VideoCapture(INPUT_FILE)
    # cap.open(0, cv2.CAP_ANY);
    if not cap.isOpened():
        print("ERROR! Unable to open camera\n")
        exit()
    #print("Webcam Video Stream open...")
    print("Press q to terminate")
    
    w = int( cap.get(3) )
    h = int( cap.get(4) )
    
    ###### Defining VideoWriter
    scale = 1
    size = ((int)(w*2*scale), (int)(h*2*scale))
    fourcc  = cv2.VideoWriter_fourcc('m','p','4','v')
    # video   = cv2.VideoWriter( 'ATC_LKPR_output.avi', fourcc, 30, size ) # fps = 30, size = ( 1024, 512 )
    result = cv2.VideoWriter(OUTPUT_FILE , fourcc , 20 , size)
    
    detector = distanceDetector()
    
    while True:
        _, frame = cap.read()
        if frame is None:
            print("ERROR! blank frame grabbed\n")
            exit()
        
        start_time = time.time()
        frame = detector.detect(frame)
        #projectedFrame = cv2.warpPerspective(frame, detector.H, (w, h), cv2.INTER_LINEAR)
        #res = np.concatenate((frame, projectedFrame), axis=1)
        total_time_taken = time.time()-start_time
        print("Total time taken: ", total_time_taken)
        print("FPS: ", 1/total_time_taken)
        cv2.imshow("Live", frame)
        result.write(frame)
        
        iKey = cv2.waitKey(1)
        if iKey == ord('q') or iKey == ord('Q'):
            break 

    result.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    # main()
    video()
