import cv2
import numpy as np
from math import *

if __name__=="__main__":
    img = cv2.imread("image.jpg")
    # cv2.imshow("image", img)
    h = np.array([[1.34262430, 2.09180830, -220.45955000], 
[0.08859860, 3.75966310, -341.46072000],
[0.00019525, 0.00417576, 1.00000000]])
    
    homography_file = cv2.FileStorage("good_homography.xml", cv2.FILE_STORAGE_READ)
    h = homography_file.getNode("homography_matrix").mat()
    homography_file.release()

    print(h)
    h = np.linalg.inv(h)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = np.zeros(img.shape, dtype=np.uint8)
    point = np.array([[0],[0],[0]])
    tpoint = np.array([[0],[0],[0]])
    
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            point[0,0] = j
            point[1,0] = i
            point[2,0] = 1
            tpoint = h@point
            tpoint = tpoint / tpoint[2,0]
            if tpoint[0,0]>=0 and tpoint[1,0]>=0 and tpoint[2,0]>=0:
                x = tpoint[1,0]
                y = tpoint[0,0]
                try:
                    dx = x-floor(x)
                    dy = y-floor(y)
                    foo = gray[floor(x),floor(y)]
                    flo = gray[ceil(x), floor(y)]
                    fol = gray[floor(x), ceil(y)]
                    fll = gray[ceil(x), ceil(y)]
                    fy = (1-dx)*foo + dx*flo
                    fy1 = (1-dx)*fol + dx*fll
                    fi = (1-dy)*fy + dy*fy1
                    #print(fi)
                    # result[i,j] = gray[round(tpoint[1,0]), round(tpoint[0,0])]
                    result[i,j] = fi
                except:
                    pass

                    
    
    cv2.imshow("gray", gray)        
    cv2.imshow("warped", result)
    
    cv2.waitKey(0)
    
