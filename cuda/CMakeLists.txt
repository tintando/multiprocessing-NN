cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_FLAGS_DEBUG "-G")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(CUDA REQUIRED)

add_executable(main main.cu src/device_info.cu)

set_property(TARGET main 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)