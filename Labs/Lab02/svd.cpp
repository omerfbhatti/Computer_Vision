#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    double adData[] = {3, 2, 4, 8, 4, 2, 1, 3, 2};
    Mat matA(3, 3, CV_64F, adData);
    cout << "A:" << endl << matA << endl;
    SVD svdA(matA, SVD::FULL_UV);
    cout << "U:" << endl << svdA.u << endl;
    cout << "W:" << endl << svdA.w << endl;
    cout << "Vt:" << endl << svdA.vt << endl;
    cout << "Reconstruction of A from its SVD:" << endl << svdA.u * Mat::diag(svdA.w) * svdA.vt << endl;
    return 0;
}

