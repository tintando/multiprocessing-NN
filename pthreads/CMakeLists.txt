cmake_minimum_required(VERSION 3.12)
project(pthreads)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g")
find_package(Threads REQUIRED)


# Add your source files here
set(SOURCES
    main.cpp
    src/threads_train.cpp
    src/activation_functions.cpp
    src/mlp.cpp
    src/dataset.cpp
)

add_executable(multithread ${SOURCES})
target_link_libraries(multithread Threads::Threads -pthread)

