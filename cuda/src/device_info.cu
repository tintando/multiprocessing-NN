#include <iostream>
#include <cuda_runtime.h>

int printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        std::cout << "Device " << i << ": " << deviceProp.name << std::endl;
        std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << "MB" << std::endl;
        std::cout << "Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << "KB" << std::endl;
        std::cout << "Shared Memory per Multiprocessor: " << deviceProp.sharedMemPerMultiprocessor / 1024 << "KB" << std::endl;
        std::cout << "Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "Max Threads per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max Threads per Multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Max Grid Size: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
        std::cout << "Max Block Size: (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << std::endl;
    }

    return 0;
}

// Device 0: NVIDIA GeForce RTX 3080
// Compute Capability: 8.6

// Total Global Memory: 10001MB
// Shared Memory per Block: 48KB
// Shared Memory per Multiprocessor: 100KB

// Multiprocessors: 68
// Max Threads per Block: 1024
// Max Threads per Multiprocessor: 1536
// Max Grid Size: (2147483647, 65535, 65535)
// Max Block Size: (1024, 1024, 64)