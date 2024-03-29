import cv2
import time

img = cv2.imread('demo_data/test2.jpeg')
 
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()
 
net = cv2.dnn.readNetFromDarknet('models/yolov4-tiny.cfg', 'models/yolov4-tiny.weights')

model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

start = time.time() 
classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)
print("time taken for inference: ", time.time()-start)

for (classId, score, box) in zip(classIds, scores, boxes):
    print("class ID: ", classId)
    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                  color=(0, 255, 0), thickness=2)
 
    text = '%s: %.2f' % (classes[classId], score)
    cv2.putText(img, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color=(0, 255, 0), thickness=2)
 
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows() 
