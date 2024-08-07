cmake_minimum_required(VERSION 2.6)

project(arcface)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/prelu.cu)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(arcface-r50-sps ${PROJECT_SOURCE_DIR}/arcface-r50-sps.cpp)
target_link_libraries(arcface-r50-sps nvinfer)
target_link_libraries(arcface-r50-sps cudart)
target_link_libraries(arcface-r50-sps myplugins)
target_link_libraries(arcface-r50-sps ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

