#include <iostream>
#include <opencv2/opencv.hpp> // This includes all of OpenCV. You could use just opencv2/highgui.hpp.

using namespace cv;           // Without this you would have to prefix every OpenCV call with cv::
using namespace std;          // Without this you would have to prefix every C++ standard library call with std::

int main(int argc, char* argv[])
{
    int iKey = -1;
    string sFilename = "../sample.jpg";
    Mat matImage = imread(sFilename);
    if (matImage.empty())
    {
        cout << "No image to show" << endl;
        return 1;
    }
    imshow("Input image", matImage);
    // Wait up to 5s for a keypress
    iKey = waitKey(5000);
    cout << "Key output value: " << iKey << endl;    
    return 0;
}
