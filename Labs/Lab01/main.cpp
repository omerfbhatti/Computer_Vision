#include <iostream>
#include <opencv2/core/core.hpp>

using namespace std;

int main(void)
{
  double adData[] = { 3, 2, 4, 8, 4, 2, 1, 3, 2 };
  cv::Mat matA( 3, 3, CV_64F, adData );
  cout << "A:" << endl << matA << endl;
  cv::SVD svdA( matA, cv::SVD::FULL_UV );
  cout << "U:" << endl << svdA.u << endl;
  cout << "W:" << endl << svdA.w << endl;
  cout << "Vt:" << endl << svdA.vt << endl;

}