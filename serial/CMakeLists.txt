cmake_minimum_required(VERSION 3.12)
project(serial)

set(CMAKE_CXX_STANDARD 14)

# Add your source files here
set(SOURCES
    main.cpp
    src/data_loading.cpp
)

add_executable(serial ${SOURCES})

