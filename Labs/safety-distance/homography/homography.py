import cv2
import numpy as np

VIDEO_FILE = "../cap2.mp4"
ROTATE = False

class sState:
    def __init__(self):
        self.pts = []
        self.matResult = None
        self.matFinal = None
        self.matPauseScreen = None
        self.point = (-1, -1)
        self.var = 0 
        self.drag = 0

def mouseHandler(event, x, y, flags, state):
    # global point, pts, var, drag, matFinal, matResult   # call global variable to use in this function
    
    # print("Var: ", state.var)
    
    if (state.var >= 4):                           # if homography points are more than 4 points, do nothing
        return
    if (event == cv2.EVENT_LBUTTONDOWN):     # When Press mouse left down
        state.drag = 1                             # Set it that the mouse is in pressing down mode
        state.matResult = state.matFinal.copy()          # copy final image to draw image
        state.point = (x, y)                       # memorize current mouse position to point var
        if (state.var >= 1):                       # if the point has been added more than 1 points, draw a line
            cv2.line(state.matResult, state.pts[state.var - 1], state.point, (0, 255, 0, 255), 2)    # draw a green line with thickness 2
        cv2.circle(state.matResult, state.point, 2, (0, 255, 0), -1, 8, 0)             # draw a current green point
        cv2.imshow("Source", state.matResult)      # show the current drawing
    if (event == cv2.EVENT_LBUTTONUP and state.drag):  # When Press mouse left up
        state.drag = 0                             # no more mouse drag
        state.pts.append(state.point)                    # add the current point to pts
        state.var += 1                             # increase point number
        state.matFinal = state.matResult.copy()          # copy the current drawing image to final image
        if (state.var >= 4):                                                      # if the homograpy points are done
            cv2.line(state.matFinal, state.pts[0], state.pts[3], (0, 255, 0, 255), 2)   # draw the last line
            cv2.fillConvexPoly(state.matFinal, np.array(state.pts, 'int32'), (0, 120, 0, 20))        # draw polygon from points
        cv2.imshow("Source", state.matFinal);
    if (state.drag):                                    # if the mouse is dragging
        state.matResult = state.matFinal.copy()               # copy final images to draw image
        state.point = (x, y)                   # memorize current mouse position to point var
        if (state.var >= 1):                            # if the point has been added more than 1 points, draw a line
            cv2.line(state.matResult, state.pts[state.var - 1], state.point, (0, 255, 0, 255), 2)    # draw a green line with thickness 2
        cv2.circle(state.matResult, state.point, 2, (0, 255, 0), -1, 8, 0)         # draw a current green point
        cv2.imshow("Source", state.matResult)           # show the current drawing

def calcHomography():
    pass

if __name__ == '__main__':
    # global matFinal, matResult, matPauseScreen         # call global variable to use in this function
    state = sState()
    key = -1;

    # --------------------- [STEP 1: Make video capture from file] ---------------------
    # Open input video file
    videoCapture = cv2.VideoCapture(VIDEO_FILE);
    if not videoCapture.isOpened():
        print("ERROR! Unable to open input video file ", VIDEO_FILE)
        sys.exit('Unable to open input video file')

    width  = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    
    fs = cv2.FileStorage('homography.yml', cv2.FILE_STORAGE_WRITE)

    # Capture loop 
    while (key < 0):        # play video until press any key
        # Get the next frame
        _, matFrameCapture = videoCapture.read()
        if matFrameCapture is None:   # no more frame capture from the video
            # End of video file
            break

        # Rotate if needed, some video has output like top go down, so we need to rotate it
        if ROTATE:
            _, matFrameDisplay = cv2.rotate(matFrameCapture, cv2.ROTATE_180)   #rotate 180 degree and put the image to matFrameDisplay
        else:
            matFrameDisplay = matFrameCapture;

        ratio = 640.0 / width
        dim = (int(width * ratio), int(height * ratio))
        # resize image to 480 * 640 for showing
        matFrameDisplay = cv2.resize(matFrameDisplay, dim)

        # Show the image in window named "robot.mp4"
        cv2.imshow(VIDEO_FILE, matFrameDisplay)
        key = cv2.waitKey(30)

        # --------------------- [STEP 2: pause the screen and show an image] ---------------------
        if (key >= 0):
            state.matPauseScreen = matFrameCapture     # transfer the current image to process
            state.matFinal = state.matPauseScreen.copy()     # copy image to final image

    # --------------------- [STEP 3: use mouse handler to select 4 points] ---------------------
    if (matFrameCapture is not None):
        state.var = 0                                             # reset number of saving points
        state.pts.clear()                                         # reset all points
        cv2.namedWindow("Source", cv2.WINDOW_AUTOSIZE)      # create a windown named source
        cv2.setMouseCallback("Source", mouseHandler, state)        # set mouse event handler "mouseHandler" at Window "Source"
        cv2.imshow("Source", state.matPauseScreen)                # Show the image
        cv2.waitKey(0)                                      # wait until press anykey
        
        print("Selected Points: ", state.pts)
        print("Var: ", state.var)
        
        if (len(state.pts) == 4):
            src = np.array(state.pts).astype(np.float32)
            scale = 80
            reals = scale * np.array([(3, 3),
                            (4, 3),
                            (4, 4),
                            (3, 4)], np.float32)

            homography_matrix = cv2.getPerspectiveTransform(src, reals);
            print("Estimated Homography Matrix is:")
            print(homography_matrix)
            fs.write('homography_matrix', homography_matrix)

            # perspective transform operation using transform matrix

            h, w, ch = state.matPauseScreen.shape
            state.matResult = cv2.warpPerspective(state.matPauseScreen, homography_matrix, (w, h), cv2.INTER_LINEAR)
            state.matPauseScreen = cv2.resize(state.matPauseScreen, dim)
            cv2.imshow("Source", state.matPauseScreen)
            state.matResult = cv2.resize(state.matResult, dim)
            cv2.imshow("Result", state.matResult)
            cv2.waitKey(0)
            
        fs.release()
        cv2.destroyWindow("Source")                         # destroy the window
    else:
        print("No pause before end of video finish. Exiting.")
