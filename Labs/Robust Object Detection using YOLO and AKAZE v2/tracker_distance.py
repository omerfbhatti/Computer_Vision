import cv2
import numpy as np
import time
from scipy.spatial import distance
from camera import cameraParameters
from homography import Homography
from tracker import Tracker


akaze_thresh:float = 3e-4 # AKAZE detection threshold set to locate about 1000 keypoints
ransac_thresh:float = 2.5 # RANSAC inlier threshold
nn_match_ratio:float = 0.8 # Nearest-neighbour matching ratio
bb_min_inliers:int = 100 # Minimal number of inliers to draw bounding box
stats_update_period:int = 10 # On-screen statistics are updated every 10 frames

class pts:
    def __init__(self, idx, d):
        self.pair = idx
        self.distance = d

class distanceDetector:
    def __init__(self):
        with open('coco.names', 'r') as f:
            self.classes = f.read().splitlines()
        
        self.P_matrix = cameraParameters("camera_parameters.yml").cameraMatrix
        self.P_matrix = np.concatenate((self.P_matrix @ np.eye(3), self.P_matrix @ np.zeros((3,1))), axis=1)
        self.P_inv = np.linalg.pinv(self.P_matrix)
        
        self.H = Homography("homography.yml").matH
        
        net = cv2.dnn.readNetFromDarknet('models/yolov4-tiny.cfg', 'models/yolov4-tiny.weights')

        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
        
        self.akaze_stats = Stats()
        
        detector = cv2.AKAZE_create()
        detector.setThreshold(akaze_thresh)
        matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
        self.tracker = Tracker(detector, matcher)
        
        self.akaze_draw_stats = None
        
        
        
    def detect(self, img):
        start = time.time() 
        classIds, scores, boxes = self.model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
        time_taken = time.time()-start
        print("time taken for inference: ", time_taken)
        
        centers = []
        test_boxes = []
        rois = []
        for (classId, score, box) in zip(classIds, scores, boxes):
            print("class ID: ", classId)
            if classId==0:
                cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                            color=(0, 255, 0), thickness=2)
            
                text = '%s: %.2f' % (self.classes[classId], score)
                cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            color=(0, 255, 0), thickness=2)
                
                #print("box: ", box)
                centerx = box[0]+box[2] / 2
                centery = box[1]+box[3] / 2
                center = (centerx, centery)
                print("center: ", center)
                centers.append(center)
                test_boxes.append(box)
                rois.append(img[box[0]:box[0]+box[2], box[1]:box[1]+box[3]])
        
        if self.tracker.first_frame is None:
            self.rois = rois
        elif self.tracker.first_frame is not None:
            self.compare_features(rois)
            
        if len(centers)>1:
            pairs = self.matchPairwiseDistances(centers)
            
        return img
    
    def compare_features(self, second_rois):
        for roi1 in self.rois:
            for roi2 in second_rois:
                
        
        
    def matchPairwiseDistances(self, centers):
        pairs = []
        for i in range(len(centers)):
            for j in range(i+1,len(centers)):
                pts1 = self.projectPoint(centers[i])
                pts2 = self.projectPoint(centers[j])
                dist_metric = distance.euclidean(pts1, pts2) * 3.28
                dist_pixel = distance.euclidean(centers[i], centers[j])
                if dist_metric < 10:
                    print("metric distance: ", dist_metric)
                    print("pixel distance: ", dist_pixel)
                    print("pair: ", i, ", ", j)
                    pairs.append(pts((i,j), dist_metric))
        return pts
    
    def projectPoint(self, x):
        return self.H @ x

    
def main():
    img = cv2.imread('demo_data/test.jpg')
    
    detector = distanceDetector()
    img = detector.detect(img)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
def video():
    cap = cv2.VideoCapture('demo_data/cap.mp4')
    # cap.open(0, cv2.CAP_ANY);
    if not cap.isOpened():
        print("ERROR! Unable to open camera\n")
        exit()
    #print("Webcam Video Stream open...")
    print("Press q to terminate")
    
    detector = distanceDetector()
    
    while True:
        _, frame = cap.read()
        if frame is None:
            print("ERROR! blank frame grabbed\n")
            exit()
        frame = detector.detect(frame)
        cv2.imshow("Live", frame)
        iKey = cv2.waitKey(1)
        if iKey == ord('q') or iKey == ord('Q'):
            break 


if __name__=="__main__":
    # main()
    video()
