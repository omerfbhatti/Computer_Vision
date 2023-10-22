### WAY !
for fname in images:
    img = cv2.imread(fname)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    res = cv2.undistort(img, mtx, dist)
    
    # crop the image
    x, y, w, h = roi
    res = res[y:y+h, x:x+w]
    
    cv2.imshow('img',img)
    cv2.imshow('res',res)
    cv2.waitKey(0)

### WAY 2

for fname in images:
    img = cv2.imread(fname)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    res = cv2.remap(img, mapx, mapy, cv.INTER_LINEAR)
    
    # crop the image
    x, y, w, h = roi
    res = res[y:y+h, x:x+w]
    
    cv2.imshow('img',img)
    cv2.imshow('res',res)
    cv2.waitKey(0)
