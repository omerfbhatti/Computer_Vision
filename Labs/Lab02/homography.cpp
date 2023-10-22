#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace cv;
using namespace std;

//#define VIDEO_FILE "../myvid1.mp4"

struct sState {
    Mat matPauseScreen, matResult, matFinal;
    Point point;
    vector<Point> pts;
    int var;
    int drag;
};

// Create mouse handler function

void mouseHandler(int, int, int, int, void*);

Mat calcHomography(struct sState *pState, cv::Mat *homography_matrix);

int main(int argc, char* argv[])
{
    Mat matFrameCapture;
    Mat matFrameDisplay;
    int key = -1;
    struct sState state;
    state.var = 0;
    state.drag = 0;

    string VIDEO_FILE = argv[1];
    
    // --------------------- [STEP 1: Make video capture from file] ---------------------
    // Open input video file
    VideoCapture videoCapture(VIDEO_FILE);
    if (!videoCapture.isOpened()) {
        cerr << "ERROR! Unable to open input video file " << VIDEO_FILE << endl;
        return -1;
    }
    
    // string filename = "../homography.xml";
    string filename = argv[2];
    FileStorage fs(filename, FileStorage::WRITE);

    fs.open(filename, FileStorage::WRITE);

    // Capture loop
    while (key < 0)        // play video until press any key
    {
        // Get the next frame
        videoCapture.read(matFrameCapture);
        if (matFrameCapture.empty()) {   // no more frame capture from the video
            // End of video file
            break;
        }
        cvtColor(matFrameCapture, matFrameCapture, COLOR_BGR2BGRA);

        // Rotate if needed, some video has output like top go down, so we need to rotate it
#if ROTATE
        rotate(matFrameCapture, matFrameCapture, RotateFlags::ROTATE_180);   //rotate 180 degree and put the image to matFrameDisplay
#endif

        float ratio = 640.0 / matFrameCapture.cols;
        resize(matFrameCapture, matFrameDisplay, cv::Size(), ratio, ratio, INTER_LINEAR);

        // Display
        imshow(VIDEO_FILE, matFrameDisplay); // Show the image in window named "robot.mp4"
        key = waitKey(30);

        // --------------------- [STEP 2: pause the screen and show an image] ---------------------
        if (key >= 0)
        {
            state.matPauseScreen = matFrameCapture;  // transfer the current image to process
            state.matFinal = state.matPauseScreen.clone(); // clone image to final image
            imwrite("image.jpg", matFrameCapture);
        }
    }

    // --------------------- [STEP 3: use mouse handler to select 4 points] ---------------------
    if (!matFrameCapture.empty())
    {
        state.var = 0;   // reset number of saving points
        state.pts.clear(); // reset all points
        namedWindow("Source", WINDOW_AUTOSIZE);  // create a windown named source
        setMouseCallback("Source", mouseHandler, &state); // set mouse event handler "mouseHandler" at Window "Source"
        imshow("Source", state.matPauseScreen); // Show the image
        waitKey(0); // wait until press anykey
        // destroyWindow("Source"); // destroy the window
        // calcHomography(&state);
        if (state.pts.size() == 4)
        {
            //Point2f src[4];
            //for (int i = 0; i < 4; i++)
            //{
            //    src[i].x = state.pts[i].x * 1.0;
            //    src[i].y = state.pts[i].y * 1.0;
            //}
            // Point2f reals[4];
            // reals[0] = Point2f(300.0, 300.0);
            // reals[1] = Point2f(400.0, 300.0);
            // reals[2] = Point2f(400.0, 400.0);
            // reals[3] = Point2f(300.0, 400.0);

            // Mat homography_matrix = getPerspectiveTransform(src, reals);
            Mat homography_matrix = Mat::zeros(1, 9, CV_32F);
            homography_matrix = calcHomography(&state, &homography_matrix);
            std::cout << "Estimated Homography Matrix is:" << std::endl;
            std::cout << homography_matrix << std::endl;
            fs << "homography_matrix" << homography_matrix;
            
            // perspective transform operation using transform matrix
            // cv::warpPerspective(state.matPauseScreen, state.matResult, homography_matrix, state.matPauseScreen.size(), cv::INTER_LINEAR);
            imshow("Source", state.matPauseScreen);
            imshow("Result", state.matResult);

            waitKey(0);
        }
    }
    else
    {
        cout << "You did not pause the screen before the video finish, the program will stop" << endl;
        return 0;
    }

    return 0;
}

// An OpenCV mouse handler function has 5 parameters: the event that occurred,
// the position of the mouse, and a user-specific pointer to an arbitrary data
// structure.

void mouseHandler(int event, int x, int y, int, void *pVoid)
{
    struct sState *pState = (struct sState *)pVoid;

    if (pState->var >= 4) // If we already have 4 points, do nothing
        return;
    if (event == EVENT_LBUTTONDOWN) // Left button down
    {
        pState->drag = 1; // Set it that the mouse is in pressing down mode
        pState->matResult = pState->matFinal.clone(); // copy final image to draw image
        pState->point = Point(x, y); // memorize current mouse position to point var
        if (pState->var >= 1) // if the point has been added more than 1 points, draw a line
        {
            line(pState->matResult, pState->pts[pState->var - 1], pState->point, Scalar(0, 255, 0, 255), 2); // draw a green line with thickness 2
        }
        circle(pState->matResult, pState->point, 2, Scalar(0, 255, 0), -1, 8, 0); // draw a current green point
        imshow("Source", pState->matResult); // show the current drawing
    }
    if (event == EVENT_LBUTTONUP && pState->drag) // When Press mouse left up
    {
        pState->drag = 0; // no more mouse drag
        pState->pts.push_back(pState->point);  // add the current point to pts
        pState->var++; // increase point number
        pState->matFinal = pState->matResult.clone(); // copy the current drawing image to final image
        if (pState->var >= 4) // if the homograpy points are done
        {
            line(pState->matFinal, pState->pts[0], pState->pts[3], Scalar(0, 255, 0, 255), 2); // draw the last line
            fillPoly(pState->matFinal, pState->pts, Scalar(0, 120, 0, 20), 8, 0); // draw polygon from points

            setMouseCallback("Source", NULL, NULL); // remove mouse event handler
        }
        imshow("Source", pState->matFinal);
    }
    if (pState->drag) // if the mouse is dragging
    {
        pState->matResult = pState->matFinal.clone(); // copy final images to draw image
        pState->point = Point(x, y); // memorize current mouse position to point var
        if (pState->var >= 1) // if the point has been added more than 1 points, draw a line
        {
            line(pState->matResult, pState->pts[pState->var - 1], pState->point, Scalar(0, 255, 0, 255), 2); // draw a green line with thickness 2
        }
        circle(pState->matResult, pState->point, 2, Scalar(0, 255, 0), -1, 8, 0); // draw a current green point
        imshow("Source", pState->matResult); // show the current drawing
    }
}

cv::Mat calcHomography(struct sState *pState, cv::Mat *homography_matrix)
{
    cout << "Calculating homography..." << endl;
    for (int i = 0; i < pState->pts.size(); i++)
    {
        cout << "Point " << i << ": " << pState->pts[i] << endl;
    }
    if (pState->pts.size() != 4) {
        cout << "Four points are needed for a homography..." << endl;
        Mat error_mat;
        return error_mat;
    }
    Mat matA = Mat::zeros(8, 9, CV_32F);
    int xprimes[] = { 300, 400, 400, 300 };
    int yprimes[] = { 300, 300, 400, 400 };
    for (int i = 0; i < pState->pts.size(); i++) {
        float x = pState->pts[i].x;
        float y = pState->pts[i].y;
        float xprime = xprimes[i];
        float yprime = yprimes[i];
        matA.at<float>(i*2, 0) = 0;
        matA.at<float>(i*2, 1) = 0;
        matA.at<float>(i*2, 2) = 0;
        matA.at<float>(i*2, 3) = -x;
        matA.at<float>(i*2, 4) = -y;
        matA.at<float>(i*2, 5) = -1;
        matA.at<float>(i*2, 6) = yprime * x;
        matA.at<float>(i*2, 7) = yprime * y;
        matA.at<float>(i*2, 8) = yprime;
        matA.at<float>(i*2+1, 0) = x;
        matA.at<float>(i*2+1, 1) = y;
        matA.at<float>(i*2+1, 2) = 1;
        matA.at<float>(i*2+1, 3) = 0;
        matA.at<float>(i*2+1, 4) = 0;
        matA.at<float>(i*2+1, 5) = 0;
        matA.at<float>(i*2+1, 6) = -xprime * x;
        matA.at<float>(i*2+1, 7) = -xprime * y;
        matA.at<float>(i*2+1, 8) = -xprime;
    }
    cout << "Matrix A:" << endl;
    cout << matA << endl;
    SVD svdA( matA, SVD::FULL_UV );
    cout << "U:" << endl << svdA.u << endl;
    cout << "W:" << endl << svdA.w << endl;
    cout << "Vt:" << endl << svdA.vt << endl;
    
    Mat matH = Mat::zeros(1, 9, CV_32F);
    for (int i=0; i<9; i++) {
        matH.at<float>(0, i) = svdA.vt.at<float>(8, i);
    }
    matH = matH / matH.at<float>(0,8);
    // cout << "H:" << endl << matH << endl;
    matH = matH.reshape(3,3); 
    //cout << "H:" << endl << matH << endl;
    /*
    Mat point = Mat::zeros(3, 1, CV_32F);
    Mat tpoint = Mat::zeros(3, 1, CV_32F);
    cv::Mat pause;
    cv::cvtColor(pState->matPauseScreen, pause, COLOR_BGR2GRAY);
    Mat result = Mat::zeros(pause.rows, pause.cols, CV_32F);
    
    for (int i=0; i< pause.rows; i++ ) {
        for (int j=0; j<pause.cols; i++) {
            point.at<float>(0,0) = j;
            point.at<float>(1,0) = i;
            point.at<float>(2,0) = 1;
            // cout << "point:" << endl << point << endl; 
            // cout << matH.type() << endl;
            // cout << point.type() << endl;
            tpoint.at<float>(0,0) = matH.at<float>(0,0) * point.at<float>(0,0) + matH.at<float>(0,1) * point.at<float>(1,0) + matH.at<float>(0,2) * point.at<float>(2,0);
            tpoint.at<float>(1,0) = matH.at<float>(1,0) * point.at<float>(0,0) + matH.at<float>(1,1) * point.at<float>(1,0) + matH.at<float>(1,2) * point.at<float>(2,0);
            tpoint.at<float>(2,0) = matH.at<float>(2,0) * point.at<float>(0,0) + matH.at<float>(2,1) * point.at<float>(1,0) + matH.at<float>(2,2) * point.at<float>(2,0);
            tpoint = tpoint/tpoint.at<float>(2,0);
            
            if (tpoint.at<float>(0,0)>=0 && tpoint.at<float>(1,0)>=0 && tpoint.at<float>(2,0)>=0) {
                result.at<float>(round(1+tpoint.at<float>(1,0)), round(1+tpoint.at<float>(0,0))) = pause.at<float>(i,j);
            }
            else {
                cout << "Error: coordinate value less than zero";
            }
        }
    }
    imshow("Warped", result);
    waitKey(0);*/    
    return matH;
}

