#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
int main()
{
    int frameAdd = 0;
    Mat frame;
    int iKey = -1;
    //--- INITIALIZE VIDEOCAPTURE
    VideoCapture cap;
    // open the default camera using default API
    // cap.open(0);
    // OR advance usage: select any API backend
    int deviceID = 0;             // 0 = open default camera
    int apiID = cv::CAP_ANY;      // 0 = autodetect default API
    // open selected camera using selected API
    cap.open(deviceID, apiID);
    // check if we succeeded
    if (!cap.isOpened()) {
        cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    //--- GRAB AND WRITE LOOP
    cout << "Start grabbing" << endl
        << "Press s to save images and q to terminate" << endl;
    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        // check if we succeeded
        if (frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        // show live and wait for a key with timeout long enough to show images
        imshow("Live", frame);
        iKey = waitKey(5);
        if (iKey == 's' || iKey == 'S')
        {
            imwrite("./images/frame" + to_string(frameAdd) + ".jpg", frame);
            wantFrame[frameAdd] = frame.clone();
            frameAdd++;
            cout << "Frame: " << frameAdd << " has been saved." << endl;
        }
        else if (iKey == 'q' || iKey == 'Q')
        {
            break;
        }
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
