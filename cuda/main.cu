#include <stdio.h>
#include <cuda.h>
#include "include/device_info.h"

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    printf("Hello World from CPU!\n");
    printDeviceInfo();
    cuda_hello<<<1,1>>>(); 
    cudaDeviceSynchronize();
    return 0;
}