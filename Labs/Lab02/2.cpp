#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

// In C++, you can define constants variable using #define
#define VIDEO_FILE "../robot.qt"
#define ROTATE false

int main(int argc, char** argv)
{
    Mat matFrameCapture;
    Mat matFrameDisplay;
    int iKey = -1;

    // Open input video file
    VideoCapture videoCapture(VIDEO_FILE);
    if (!videoCapture.isOpened()) {
        cerr << "ERROR! Unable to open input video file " << VIDEO_FILE << endl;
        return -1;
    }

    namedWindow(VIDEO_FILE, WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
    int n_frame = 0;

    // Capture loop
    while (true)        // play video until user presses <space>
    {
        // Get the next frame
        videoCapture.read(matFrameCapture);
        if (matFrameCapture.empty())
        {
            // End of video file
            break;
        }

        // We can rotate the image easily if needed.
#if ROTATE
        rotate(matFrameCapture, matFrameDisplay, RotateFlags::ROTATE_180);   //rotate 180 degree and put the image to matFrameDisplay
#else
        matFrameDisplay = matFrameCapture;
#endif

        float ratio = 480.0 / matFrameDisplay.rows;
        resize(matFrameDisplay, matFrameDisplay, cv::Size(), ratio, ratio, INTER_LINEAR); // resize image to 480p for showing

        // Display
        imshow(VIDEO_FILE, matFrameDisplay); // Show the image in window named "robot.mp4"
        n_frame++;
        displayOverlay(VIDEO_FILE, "Frame #"+to_string(n_frame));

        iKey = -1;
        while (iKey!=int(' ') && iKey!=int('q') ) {
            iKey = waitKey(30); // Wait 30 ms to give a realistic playback speed
            if (getWindowProperty(VIDEO_FILE, WND_PROP_VISIBLE)==0) {
                   iKey = int('q');
            }
        }
        if (iKey==int('q')) {
            break;
        }

    }
    return 0;
}
