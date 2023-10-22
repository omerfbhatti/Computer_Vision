#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
       // string filename = "../myvid1.mp4"; // Name of video file to open
       string filename = argv[1];
       VideoCapture cap(filename);     // declaring a capture object to grab video frames
       if (!cap.isOpened()) {         // check if capture object can access the file
            cerr << "Unable to open file!" << endl;
            return 0;
       }
       
       string homography_file = "../homography.xml";
       FileStorage fs(homography_file, FileStorage::READ);
       fs.open(homography_file, FileStorage::READ);
       
       Mat homography_matrix; 
       fs["homography_matrix"] >> homography_matrix;
       cout << "homography_matrix:" << endl << homography_matrix << endl;
       
       // Create some random colors
       vector<Scalar> colors;
       RNG rng;
       for (int i=0; i<100; i++) {
            int r = rng.uniform(0,256);
            int g = rng.uniform(0,256);
            int b = rng.uniform(0,256);
            colors.push_back(Scalar(r,g,b));
       }

       // GET IMAGE FROM VIDEO AND PROCESS TO GET FEATURES TO TRACK
       Mat old_frame, old_gray; // Declare matrices to hold previous frames (color and grayscale)
       vector<Point2f> p0, p1;  // Declare two point vectors (float dtype)
       cap >> old_frame; // Copy from video stream to old frame
       cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);  // convert to grayscale
       goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);  // get corner features to track from the first frame 
       Mat mask = Mat::zeros(old_frame.size(), old_frame.type());  // declare a zero matrix of same size as the frame --> as a mask
       
       // WARP THE IMAGE AND PROCESS IT TO GET FEATURES TO TRACK
       Mat old_warped, old_warped_gray;
       vector<Point2f> rp0, rp1;
       cv::warpPerspective(old_frame, old_warped, homography_matrix, old_frame.size(), cv::INTER_LINEAR);
       cvtColor(old_warped, old_warped_gray, COLOR_BGR2GRAY);
       goodFeaturesToTrack(old_warped_gray, rp0, 100, 0.3, 7, Mat(), 7, false, 0.04); 
       Mat mask_warped = Mat::zeros(old_warped.size(), old_warped.type());
       
       // VIDEO WRITER TO SAVE THE OPTICAL FLOW VIDEO
       VideoWriter writer;
       int codec = VideoWriter::fourcc('M', 'P', '4', 'V');  
       double fps = 30.0;
       // string saveFilename = "../opticalFlow2.mp4";
       string saveFilename = argv[2];
       Size sizeFrame(int(old_frame.rows/2), int(old_frame.cols/2));
       writer.open(saveFilename, codec, fps, sizeFrame, true);

       while(true) {
            Mat frame, frame_gray;  // Declare current frame for every loop
            Mat warped, warped_gray;
            
            cap >> frame;           // Get current frame from the video stream
            if (frame.empty()) {   // Check if frame is empty
                break;
            }

            cvtColor(frame, frame_gray, COLOR_BGR2GRAY);  // convert to grayscale
            cv::warpPerspective(frame, warped, homography_matrix, frame.size(), cv::INTER_LINEAR);
            cvtColor(warped, warped_gray, COLOR_BGR2GRAY);
            

            vector<uchar> status, status_warped;
            vector<float> err, err_warped;     // Error vector
            TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
            
            // Calculate optical flow between successive frames
            try {
                calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15,15), 2, criteria);
            }   
            catch (...) {
                old_gray = frame_gray.clone();
                old_warped_gray = warped_gray.clone();
                goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
                goodFeaturesToTrack(old_warped_gray, rp0, 100, 0.3, 7, Mat(), 7, false, 0.04); 
                continue;
            }
            // calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15,15), 2, criteria);
            /*
            try {
                calcOpticalFlowPyrLK(old_warped_gray, warped_gray, rp0, rp1, status_warped, err_warped, Size(15,15), 2, criteria);
            }   
            catch (...) {
                old_gray = frame_gray.clone();
                old_warped_gray = warped_gray.clone();
                goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
                goodFeaturesToTrack(old_warped_gray, rp0, 100, 0.3, 7, Mat(), 7, false, 0.04); 
                continue;
            }*/
            calcOpticalFlowPyrLK(old_warped_gray, warped_gray, rp0, rp1, status_warped, err_warped, Size(15,15), 2, criteria);

            vector<Point2f> good_new;
            for (uint i=0; i < p0.size(); i++) {
                if (status[i]==1) {
                    good_new.push_back(p1[i]);
                    line(mask, p1[i], p0[i], colors[i], 2);
                    circle(frame, p1[i], 5, colors[i], -1);
                }
            }
            
            vector<Point2f> good_new_warped;
            for (uint i=0; i < rp0.size(); i++) {
                if (status_warped[i]==1) {
                    good_new_warped.push_back(rp1[i]);
                    line(mask_warped, rp1[i], rp0[i], colors[i], 2);
                    circle(warped, rp1[i], 5, colors[i], -1);
                }
            }

            Mat img, warped_img;
            add(frame, mask, img);
            add(warped, mask_warped, warped_img);
            
            int rows = max(img.rows, warped_img.rows);
            int cols = img.cols + warped_img.cols;
            
            Mat combined_Matrix(rows, cols, img.type());
            img.copyTo(combined_Matrix(Rect(0, 0, img.cols, img.rows)));
            warped_img.copyTo(combined_Matrix(Rect(img.cols, 0, warped_img.cols, warped_img.rows)));
            
            // imshow("Frame", img);
            // imshow("Warped Frame", warped_img);
            Mat resized_result;
            resize(combined_Matrix, resized_result, sizeFrame);
            writer.write(resized_result);

            imshow("Video", resized_result);
            
            int keyboard = waitKey(30);
            if (keyboard == 'q' || keyboard == 27) {
                break;
            }

            old_gray = frame_gray.clone();
            old_warped_gray = warped_gray.clone();
            p0 = good_new;
            rp0 = good_new_warped;
      }
      
      writer.release();
      cap.release();
}
