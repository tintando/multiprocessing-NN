#include <stdio.h>
#include <cuda.h>
#include "include/device_info.h"
#include <unistd.h>
#include <cuda_runtime.h>
#include "include/data_loading.h"

void checkCudaError(cudaError_t error){
    if (error != cudaSuccess) {
        printf("Error: %s:%d, ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}

__global__ void cuda_hello(){
    printf("Block ID: (%d, %d, %d), Thread ID: (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);


int main() {
    printf("Hello World from CPU!\n");
    printDeviceInfo();
    char* filename = "/home/tintando/Documents/multiprocessing-NN/cuda/datasets/california.csv";
    Sample* samples = readDataset(filename);
    printSamples(samples, 5);


    cuda_hello<<<1,1>>>(); 
    cudaDeviceSynchronize();
    return 0;
}