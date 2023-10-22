import cv2
import numpy as np
from typing import List #use it for :List[...]

import cv2
import numpy as np
from typing import List #use it for :List[...]

class Homography:
    matH = np.zeros((3, 3))
    widthOut : int
    heightOut : int
    cPoints : int
    aPoints:list = []

    def __init__(self, homography_file = None):
        self.cPoints = 0
        if homography_file is not None:
            self.read(homography_file)

    def read(self, homography_file):
        fileStorage = cv2.FileStorage(homography_file, cv2.FILE_STORAGE_READ)
        if not fileStorage.isOpened():
            return False

        self.cPoints = 0
        for i in range(4):
            points = fileStorage.getNode("aPoints" + str(i))
            self.aPoints.append(points.mat())
            self.cPoints += 1
        self.matH = fileStorage.getNode("matH").mat()
        self.widthOut = int(fileStorage.getNode("widthOut").real())
        self.heightOut = int(fileStorage.getNode("heightOut").real())
        fileStorage.release()
        return True

    def write(self, homography_file):
        fileStorage = cv2.FileStorage(homography_file, cv2.FILE_STORAGE_WRITE)
        if not fileStorage.isOpened():
            return False

        for i in range(4):
            fileStorage.write("aPoints" + str(i), self.aPoints[i])

        fileStorage.write("matH", self.matH)
        fileStorage.write("widthOut", self.widthOut)
        fileStorage.write("heightOut", self.heightOut)
        fileStorage.release()
        return True


class tsHomographyData:
    matH = np.zeros((3, 3))
    widthOut : int
    heightOut : int
    cPoints : int
    aPoints:list = []


def readHomography(homography_file: str) -> "tsHomographyData":
    fileStorage = cv2.FileStorage(homography_file, cv2.FILE_STORAGE_READ)
    if not fileStorage.isOpened():
        return None

    
    pHomographyData.cPoints = 0
    for i in range(points.size()):
        points = fileStorage.getNode("aPoints" + str(i))
        pHomographyData.aPoints.append(points.mat())
        pHomographyData.cPoints += 1
    pHomographyData.matH = fileStorage.getNode("matH").mat()
    pHomographyData.widthOut = int(fileStorage.getNode("widthOut").real())
    pHomographyData.heightOut = int(fileStorage.getNode("heightOut").real())
    fileStorage.release()
    return pHomographyData

def writeHomography(homography_file: str, pHomographyData:"tsHomographyData") -> bool:
    fileStorage = cv2.FileStorage(homography_file, cv2.FILE_STORAGE_WRITE)
    if not fileStorage.isOpened():
        return False

    for i in range(4):
        fileStorage.write("aPoints" + str(i), pHomographyData.aPoints[i])

    fileStorage.write("matH", pHomographyData.matH)
    fileStorage.write("widthOut", pHomographyData.widthOut)
    fileStorage.write("heightOut", pHomographyData.heightOut)
    fileStorage.release()
    return True
