#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob

def findObjectAndImagePoints(images):
    # Defining the dimensions of checkerboard
    
    # CHECKERBOARD = (8,11)
    CHECKERBOARD = (6,9)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 


    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    prev_img_shape = None
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
        cv2.imshow('img', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    return objpoints, imgpoints
    
    
def calibrateAndSave(images, objpoints, imgpoints, camera_settings_file):
    img = cv2.imread(images[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h,w = img.shape[:2]

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)

    saveCameraMatrix(h, w, mtx, dist, rvecs, tvecs, camera_settings_file)
    findReprojectionError(objpoints, imgpoints, h, w, mtx, dist, rvecs, tvecs)
    
def saveCameraMatrix(h, w, mtx, dist, rvecs, tvecs, camera_settings_file):
    fs = cv2.FileStorage(camera_settings_file, cv2.FILE_STORAGE_WRITE)
    fs.write("image_height", h)
    fs.write("image_width", w)
    fs.write("Camera_matrix", mtx)
    fs.write("dist", dist)
    fs.write("rvecs", np.array([rvecs]))
    fs.write("tvecs", np.array([tvecs]))
    fs.release()

def findReprojectionError(objpoints, imgpoints, h, w, mtx, dist, rvecs, tvecs):
    ### FIND REPROJECTION ERROR
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total error: {}".format(mean_error/len(objpoints)) )
    
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
        
def openCalibrationSettings(filename):

    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    try:
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

def openHomographySettings(filename):
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    try:
        h = fs.getNode("homography_matrix").mat()
        print("Homography matrix : \n")
        print(h)
        fs.release()
        return h
    except:
        print("Error occured in reading file.")
        raise ValueError()
    
    
if __name__=="__main__":
    calibration_images_dir_name = "sample-calib-images-jetson-rpicam" 
    camera_settings_file = "rpi_camera_parameters.yml"
    VIDEO_FILE = "robot.qt"
    homography_file = 'homography.yml'
    homography_distorted_file = 'homography_distorted.yml'
    
    try:
        h, w, mtx, dist, rvecs, tvecs = openCalibrationSettings(camera_settings_file)
        print("Successfully loaded camera settings.")
    except:
        # Extracting path of individual image stored in a given directory
        images = glob.glob('./'+ calibration_images_dir_name +'/*.jpg')
        print("Camera matrix file not found. Proceeding with calibration.")
        objpoints, imgpoints = findObjectAndImagePoints(images)
        calibrateAndSave(images, objpoints, imgpoints, camera_settings_file)
        print("Finished camera calibration.")
        h, w, mtx, dist, rvecs, tvecs = openCalibrationSettings(camera_settings_file)
        
    try:
        homography_matrix = openHomographySettings(homography_file)
        homography_matrix_distorted = openHomographySettings(homography_distorted_file)
       
    except:
        sys.exit("Error opening Homography files")
        
    
    videoCapture = cv2.VideoCapture(VIDEO_FILE);
    if not videoCapture.isOpened():
        sys.exit(f"ERROR! Unable to open input video file {VIDEO_FILE}")
        
    width  = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    ratio = 640.0 / width
    dim = (int(width * ratio), int(height * ratio))
    
    frame_size = (dim[0]*2, dim[1]*2)
    # frame_size = (720, 480)
    print("frame_size: ", frame_size)
    OUTPUT_FILE = "rectified_video.mp4"
    vidWriter = cv2.VideoWriter(OUTPUT_FILE, 
                         cv2.VideoWriter_fourcc('m','p','4','v'),
                         15, frame_size)
    
    # Capture loop
    key = -1
    while (key != ord('q')):        # play video until press any key
        # Get the next frame
        _, img = videoCapture.read()
        if img is None:   # no more frame capture from the video
            # End of video file
            break
        
        undistorted_img = showUndistorted(img, mtx, dist)
        
        h, w, ch = img.shape
        rec = cv2.warpPerspective(img, homography_matrix_distorted, (w, h), cv2.INTER_LINEAR)
        img = cv2.resize(img, dim)
        img = cv2.putText(img = img, text = "Original Image", org = (10, 20),
                          fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.5, color = (125, 246, 55), thickness = 2)
        rec = cv2.resize(rec, dim)
        res = cv2.putText(img = rec, text = "Rectified Image / Projective Transform", org = (10, 20),
                          fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.5, color = (125, 246, 55), thickness = 2)
        img = np.concatenate((img, rec), axis=1)
        
        h, w, ch = undistorted_img.shape
        res = cv2.warpPerspective(undistorted_img, homography_matrix, (w, h), cv2.INTER_LINEAR)
        undistorted_img = cv2.resize(undistorted_img, dim)
        undistorted_img = cv2.putText(img = undistorted_img, text = "undistorted_img", org = (10, 20),
                                fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.5, color = (125, 246, 55), thickness = 2)
        
        res = cv2.resize(res, dim)
        res = cv2.putText(img = res, text = "Projective Transform from Undistorted Image", org = (10, 20),
                                fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.5, color = (125, 246, 55), thickness = 2)
        res = np.concatenate((undistorted_img, res), axis=1)
        
        res = np.concatenate((img, res), axis=0)
        res = cv2.resize(res, frame_size)
        # print(res.shape)
        
        #cv2.imshow('original_img', img)
        vidWriter.write(res)
        cv2.imshow('res', res)
        key = cv2.waitKey(10)
        
    vidWriter.release()
    videoCapture.release()
    cv2.destroyAllWindows()
    # for fname in images:
    #     img = cv2.imread(fname)
    #     undistorted_img, w, h = showUndistorted(img, mtx, dist)
    #     cv2.imshow('img', img)
    #     cv2.imshow('undistorted_img', undistorted_img)
    #     cv2.waitKey(0)
    #     res = cv2.warpPerspective(undistorted_img, homography_matrix, (w, h), cv2.INTER_LINEAR)
    #     cv2.imshow('res', res)
    #     cv2.waitKey(0)
        
    # objpoints, imgpoints = findObjectAndImagePoints(images)
    # findReprojectionError(objpoints, imgpoints, h, w, mtx, dist, rvecs, tvecs) 

    
