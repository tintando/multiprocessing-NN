cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(CUDA REQUIRED)

add_executable(cuda_project main.cu src/device_info.cu)

set_property(TARGET cuda_project 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)