cmake_minimum_required(VERSION 2.8)
project(UtilsCtrlDemo)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES ./UtilsCtrlDemo.cpp)


#link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")    # import info with openmp if find openmp
#include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp  -lomp")

add_executable(UtilsCtrlDemo ${SOURCE_FILES})