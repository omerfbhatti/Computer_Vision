cmake_minimum_required(VERSION 3.12)

project(Lab02)
set(CMAKE_CXX_STANDARD 14)

add_executable(opticalFlow opticalFlow.cpp)
add_executable(homo homography.cpp)

find_package(OpenCV 4.5.5 REQUIRED PATHS /usr/local/opencv-4.5.5/)
target_include_directories(opticalFlow PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opticalFlow ${OpenCV_LIBS})
target_include_directories(homo PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(homo ${OpenCV_LIBS})
