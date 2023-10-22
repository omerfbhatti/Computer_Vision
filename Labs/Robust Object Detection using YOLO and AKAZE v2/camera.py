import cv2
import numpy as np

class cameraParameters:
    def __init__(self, camera_parameters_file):
        self.cameraMatrix = []
        self.h = 0
        self.w = 0
        self.dist_coeff = []
        self.rvecs = []
        self.tvecs = []
        print("=============================")
        print("Reading camera Parameters file...")
        self.read(camera_parameters_file)
        print("[OK]")
        print("=============================")
        
    def read(self, camera_parameters_file):
        fileStorage = cv2.FileStorage(camera_parameters_file, cv2.FILE_STORAGE_READ)
        if not fileStorage.isOpened():
            return False

        self.cameraMatrix = fileStorage.getNode("Camera_matrix").mat()
        self.dist_coeff = fileStorage.getNode("dist").mat()
        self.h = int(fileStorage.getNode("image_height").real())
        self.w = int(fileStorage.getNode("image_width").real())
        self.rvecs = fileStorage.getNode("rvecs").mat().squeeze(0)
        self.tvecs = fileStorage.getNode("tvecs").mat().squeeze(0)
        print("P:")
        print(self.cameraMatrix)
        fileStorage.release()
        return True 
