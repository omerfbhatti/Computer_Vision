cmake_minimum_required(VERSION 3.12)

project(hello)
set(CMAKE_CXX_STANDARD 14)

add_executable(hello main.cpp)
find_package(OpenCV 4.5.5 REQUIRED PATHS /usr/local/opencv-4.5.5/)
target_include_directories(hello PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hello ${OpenCV_LIBS})

