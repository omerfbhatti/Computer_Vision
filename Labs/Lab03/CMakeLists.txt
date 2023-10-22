cmake_minimum_required(VERSION 3.12)

project(Lab02)
set(CMAKE_CXX_STANDARD 14)

add_executable(calibration camera_calibration.cpp)

find_package(OpenCV 4.5.5 REQUIRED PATHS /usr/local/opencv-4.5.5/)
target_include_directories(calibration PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(calibration ${OpenCV_LIBS})
