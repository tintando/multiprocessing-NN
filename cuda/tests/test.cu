#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define K 3
__global__ void matrixOuterProductKernel(float* A, float* B, float* C) {
    // Calculate the row and column indices of the current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Allocate shared memory for storing the relevant vectors and result fragment
    __shared__ float sA[K][K];
    __shared__ float sB[K][K];
    __shared__ float sC[K][K];
    
    // Initialize the result fragment to zero
    sC[threadIdx.y][threadIdx.x] = 0.0;
    
    // Loop over the necessary vectors and accumulate the result fragment
    for (int i = 0; i < K; i++) {
        // Load the relevant elements of the vectors into shared memory
        sA[threadIdx.y][i] = A[row * K + i];
        sB[i][threadIdx.x] = B[i * K + col];
        
        // Wait for all threads to finish loading the vectors
        __syncthreads();
        
        // Calculate the outer product and add it to the result fragment
        for (int j = 0; j < K; j++) {
            sC[threadIdx.y][threadIdx.x] += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
        
        // Wait for all threads to finish the current iteration before loading the next vectors
        __syncthreads();
    }
    
    // Write the result fragment back to global memory
    C[row * K + col] = sC[threadIdx.y][threadIdx.x];
}


int main(int argc, char* argv[]) {
    // Allocate memory for the input and output matrices
    int N = 3;
    float* A = (float*)malloc(N * N * sizeof(float));
    float* B = (float*)malloc(N * N * sizeof(float));
    float* C = (float*)malloc(N * N * sizeof(float));
    float* dA;
    float* dB;
    float* dC;
    cudaMalloc((void**)&dA, N * N * sizeof(float));
    cudaMalloc((void**)&dB, N * N * sizeof(float));
    cudaMalloc((void**)&dC, N * N * sizeof(float));
    
    // Initialize the input matrices
    for (int i = 0; i < N * N; i++) {
        B[i] = i + 1;
    }
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = B[j * N + i];
        }
    }
    
    // Print matrix A
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", A[i * N + j]);
        }
        printf("\n");
    }
    
    // Print matrix B
    printf("Matrix B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", B[i * N + j]);
        }
        printf("\n");
    }
    
    
    // Copy the input matrices to the device
    cudaMemcpy(dA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    
    // Launch the kernel to compute the outer product
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    matrixOuterProductKernel<<<gridDim, blockDim>>>(dA, dB, dC);
    
    // Copy the result matrix back to the host
    cudaMemcpy(C, dC, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print the result matrix
    printf("Matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", C[i * N + j]);
        }
        printf("\n");
    }
    // Free device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    
    // Free host memory
    free(A);
    free(B);
    free(C);
    
    return 0;
}