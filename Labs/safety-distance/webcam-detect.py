import cv2
import numpy as np
import time
from scipy.spatial import distance
from camera import cameraParameters
from homography import Homography


akaze_thresh:float = 3e-4 # AKAZE detection threshold set to locate about 1000 keypoints
ransac_thresh:float = 2.5 # RANSAC inlier threshold
nn_match_ratio:float = 0.8 # Nearest-neighbour matching ratio
bb_min_inliers:int = 100 # Minimal number of inliers to draw bounding box
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
        
        # self.P_matrix = cameraParameters("camera_parameters.yml").cameraMatrix
        # self.P_matrix = np.concatenate((self.P_matrix @ np.eye(3), self.P_matrix @ np.zeros((3,1))), axis=1)
        # self.P_inv = np.linalg.pinv(self.P_matrix)
        
        self.H = np.array([[ 2.45404120e+00,  2.50349395e+00, -8.94422410e+02],
                           [-6.78149193e-01,  7.29390278e+00, -1.81510214e+03],
                           [-3.40011893e-04,  5.53516414e-03,  1.00000000e+00]])       #Homography("homography.yml").matH
        
        net = cv2.dnn.readNetFromDarknet('models/yolov4-tiny.cfg', 'models/yolov4-tiny.weights')

        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
        
        
    def detect(self, img):
        start = time.time() 
        classIds, scores, boxes = self.model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
        time_taken = time.time()-start
        print("time taken for inference: ", time_taken)
        
        centers = []
        test_boxes = []
        rois = []
        for (classId, score, box) in zip(classIds, scores, boxes):
            # print("class ID: ", classId)
            if classId==0:
                #cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                #            color=(0, 255, 0), thickness=2)
            
                #text = '%s: %.2f' % (self.classes[classId], score)
                #cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #            color=(0, 255, 0), thickness=2)
                
                #print("box: ", box)
                centerx = box[0]+box[2] / 2
                centery = box[1]+box[3]
                center = (centerx, centery)
                # print("center: ", center)
                centers.append(center)
                test_boxes.append(box)
                rois.append(img[box[0]:box[0]+box[2], box[1]:box[1]+box[3]])
        
        if len(centers)>1:
            pairs = self.matchPairwiseDistances(centers)
        
            for match in pairs:
                box_1 = boxes[match.p1_idx]
                box_2 = boxes[match.p2_idx]
                center_1 = (int(centers[match.p1_idx][0]), int(centers[match.p1_idx][1])) 
                center_2 = (int(centers[match.p2_idx][0]), int(centers[match.p2_idx][1]))
                
                cv2.rectangle(img, (box_1[0], box_1[1]), (box_1[0] + box_1[2], box_1[1] + box_1[3]),
                                color=(0, 0, 255), thickness=2)
                
                text = '%s: %.2f' % ("distance: ", match.distance)
                cv2.putText(img, text, (box_1[0], box_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color=(0, 0, 255), thickness=2)
                
                cv2.rectangle(img, (box_2[0], box_2[1]), (box_2[0] + box_2[2], box_2[1] + box_2[3]),
                                color=(0, 0, 255), thickness=2)
                
                text = '%s: %.2f' % ("distance: ", match.distance)
                cv2.putText(img, text, (box_2[0], box_2[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                color=(0, 0, 255), thickness=2)        
                
                print("center: ", center_1)
                cv2.line(img, center_1, center_2, (0, 0, 255), 2)
            
        return img
    
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
    
    OUTPUT_FILE = "YOLO_out.mp4"
    
    cap = cv2.VideoCapture('cap3.mp4')
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
    size = ((int)(w*2*scale), (int)(h*scale))
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
        projectedFrame = cv2.warpPerspective(frame, detector.H, (w, h), cv2.INTER_LINEAR)
        res = np.concatenate((frame, projectedFrame), axis=1)
        total_time_taken = time.time()-start_time
        print("Total time taken: ", total_time_taken)
        print("FPS: ", 1/total_time_taken)
        cv2.imshow("Live", res)
        result.write(res)
        
        iKey = cv2.waitKey(1)
        if iKey == ord('q') or iKey == ord('Q'):
            break 

    result.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    # main()
    video()
