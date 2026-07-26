cmake_minimum_required(VERSION 3.10)
project(yolo_cpp_demo)

find_package(OpenCV REQUIRED)

add_executable(yolo_cpp main.cpp)
target_link_libraries(yolo_cpp ${OpenCV_LIBS})
