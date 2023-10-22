import cv2
import numpy as np
import os
import glob

OUTPUT_FILE = "cap3.mp4"

if __name__=="__main__":
    cap = cv2.VideoCapture()
    cap.open(2, cv2.CAP_ANY);
    if not cap.isOpened():
        print("ERROR! Unable to open camera\n")
        exit()

    width = cap.get(3)
    height = cap.get(4)
    
    ###### Defining VideoWriter
    scale = 1
    size = ((int)(width*scale), (int)(height*scale))
    fourcc  = cv2.VideoWriter_fourcc('m','p','4','v')
    # video   = cv2.VideoWriter( 'ATC_LKPR_output.avi', fourcc, 30, size ) # fps = 30, size = ( 1024, 512 )
    result = cv2.VideoWriter(OUTPUT_FILE , fourcc , 20 , size)

    while True:
        _, frame = cap.read()
        if frame is None:
            print("ERROR! blank frame grabbed\n")
            exit()
        cv2.imshow("Live", frame)
        result.write(frame)
        iKey = cv2.waitKey(10)
        if iKey == ord('q') or iKey == ord('Q'):
            break 
