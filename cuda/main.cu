#include <stdio.h>
#include <cuda.h>
#include "include/device_info.h"
#include <unistd.h>

__global__ void cuda_hello(){
    int a = 24;
    printf("Hello World from GPU!\n");
    printf("a = %d\n", a);
    printf("blockIdx.x = %d\n", blockIdx.x);
}

int main() {
    printf("Hello World from CPU!\n");
    // sleep(100);
    printDeviceInfo();
    cuda_hello<<<1,1>>>(); 
    cudaDeviceSynchronize();
    return 0;
}