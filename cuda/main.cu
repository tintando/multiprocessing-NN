#include <stdio.h>
#include <cuda.h>
#include "include/device_info.h"
#include <unistd.h>
#include <cuda_runtime.h>
#include "include/data_loading.h"
#include <curand_kernel.h>

#define N_FEATURES 8
#define N_LABELS 1
#define TILE_SIZE 16

struct MLP {
    int n_layers;
    int* layers; //array of hidden layer sizes
    float** weights; //array of pointers to arrays of floats (linearized matrices)
    float** biases; //array of pointers to arrays of floats
    float** activations; //array of pointers to arrays of floats
    float** logits; //array of pointers to arrays of floats
    float** gradients; //array of pointers to arrays of floats (linearizzed matrices)
    float** deltas; //array of pointers to arrays of floats
    float* inputs; //array of floats
    float* labels; //array of floats
};

void checkCudaError(cudaError_t error){
    if (error != cudaSuccess) {
        printf("Error: %s:%d, ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}

__global__ void cuda_hello(){
    printf("Block ID: (%d, %d, %d), Thread ID: (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
};


__global__ void gpu_print_features(float* features){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 4){
        printf("Sample %d: ", idx);
        for(int i = 0; i < 8; i++){
            printf("%f ", features[idx * 8 + i]);
        }
        printf("\n");
    }
}

__global__ void curand_init(curandState* state, unsigned long seed){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0){
        curand_init(seed, idx, 0, state);
    }
}
__global__ void d_initializeWeights(float* weights, int n_values, float range, curandState* state){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n_values){
        weights[idx] = 0.0;
    }    
}

float** d_allocateWeightShaped(int* layers, int n_layers){
    float** d_weights;
    checkCudaError(cudaMalloc((void**)&d_weights, (n_layers + 1) * sizeof(float*)));
    float** weights = (float**)malloc((n_layers + 1) * sizeof(float*));
    for(int i = 0; i < n_layers + 1; i++) {
        int n_rows = i == 0 ? N_FEATURES : layers[i - 1];
        int n_cols = i == n_layers ? N_LABELS : layers[i];
        // printf("Weights %d: %d x %d\n", i, n_rows, n_cols);
        checkCudaError(cudaMalloc(&weights[i], n_rows * n_cols * sizeof(float)));
        if(weights[i] == NULL) {
            printf("Memory allocation failed.\n");
            return NULL;
        }
    }
    cudaMemcpy(d_weights, weights, (n_layers + 1) * sizeof(float*), cudaMemcpyHostToDevice);
    return d_weights;
}

float** d_allocateBiasShaped(int* layers, int n_layers){
    float** d_biases;
    checkCudaError(cudaMalloc((void**)&d_biases, n_layers * sizeof(float*)));
    float** biases = (float**)malloc(n_layers * sizeof(float*));
    for(int i = 0; i < n_layers; i++) {
        int n_cols = layers[i];
        // printf("Biases %d: %d\n", i, n_cols);
        checkCudaError(cudaMalloc(&biases[i], n_cols * sizeof(float)));
    }
    cudaMemcpy(d_biases, biases, n_layers * sizeof(float*), cudaMemcpyHostToDevice);
    return d_biases;
}


int main(int argc, char* argv[]) {
    //usage ./main hidden_size1 hidden_size2 ... hidden_sizeN epochs batch_size
   
    printf("Hello World from CPU!\n");
    printDeviceInfo();

    if(argc < 4) {
        printf("Invalid number of arguments. Please provide at least 3 arguments.\n");
        return 1;
    }

    // Allocate memory for the integer array that will store the hidden layer sizes
    int n_hidden_layers = argc - 3;
    int* layers = (int*)malloc(n_hidden_layers * sizeof(int));
    if(layers == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    // Convert arguments to integers and store them in the array
    for(int i = 1; i < argc-2; i++) {
        layers[i - 1] = atoi(argv[i]);
    }
    int epochs = atoi(argv[argc-2]);
    int batch_size = atoi(argv[argc-1]);

    printf("Hidden layer sizes: ");
    for(int i = 0; i < argc-2; i++) {
        printf("%d ", layers[i]);
    }
    
    printf("\nEpochs: %d\n", epochs);
    printf("Batch size: %d\n\n", batch_size);


    // Load the dataset (must have 8 features and 1 target)
    char* filename = "/home/tintando/Documents/multiprocessing-NN/cuda/datasets/california.csv";

    // Read the dataset
    int n_samples;
    Sample* samples = readDataset(filename, &n_samples);
    printf("Number of samples: %d\n", n_samples);
    printSamples(samples, 5);

    //memory needed 
    //weights 8xhiddensize1 + hiddensize1xhiddensize2 + ... + hiddensize2x1 
    //biases hiddensize1 + hiddensize2 + ... + 1
    //activations batch_size x hiddensize1 + batch_size x hiddensize2 + ... + batch_size x 1
    //deltas batch_size x hiddensize1 + batch_size x hiddensize2 + ... + batch_size x 1
    //outputs batch_size x hiddensize1 + batch_size x hiddensize2 + ... + batch_size x 1
    //inputs batch_size x 8
    //targets batch_size x 1


    MLP mlp;


    // coalesce features and labels
    float* h_features = (float*)malloc(n_samples * N_FEATURES * sizeof(float));
    float* h_labels = (float*)malloc(n_samples * N_LABELS * sizeof(float));
    for(int i = 0; i < n_samples; i++) {
        memcpy(h_features + i * N_FEATURES, samples[i].features, N_FEATURES * sizeof(float));
        h_labels[i] = samples[i].label;
    }
    
    // Allocate memory for the features and labels on the device and copy
    float* d_features;
    float* d_labels;
    checkCudaError(cudaMalloc((void**)&d_features, n_samples * N_FEATURES * sizeof(float)));
    cudaMemcpy(d_features, h_features, n_samples * N_FEATURES * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(cudaMalloc((void**)&d_labels, n_samples * N_LABELS * sizeof(float)));
    cudaMemcpy(d_labels, h_labels, n_samples * N_LABELS * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for the weights, biases, activations, logits, gradients, deltas

    // Allocate memory for the weights
    float** d_weights = d_allocateWeightShaped(layers, n_hidden_layers); //array of pointers to device memory

    // Allocate memory for the biases
    float** d_biases = d_allocateBiasShaped(layers, n_hidden_layers); //array of pointers to device memory

    // Allocate memory for the activations
    float** d_activations = d_allocateBiasShaped(layers, n_hidden_layers); //array of pointers to device memory

    // Allocate memory for the logits
    float** d_logits = d_allocateBiasShaped(layers, n_hidden_layers); //array of pointers to device memory

    // Allocate memory for the gradients
    float** d_gradients = d_allocateWeightShaped(layers, n_hidden_layers); //array of pointers to device memory

    // Allocate memory for the deltas
    float** d_deltas = d_allocateBiasShaped(layers, n_hidden_layers); //array of pointers to device memory

    mlp.n_layers = n_hidden_layers;
    mlp.layers = layers;
    mlp.weights = d_weights;
    mlp.biases = d_biases;
    mlp.activations = d_activations;
    mlp.logits = d_logits;
    mlp.gradients = d_gradients;
    mlp.deltas = d_deltas;
    mlp.inputs = d_features;
    mlp.labels = d_labels;

    // Initialize curand
    curandState* d_state;
    checkCudaError(cudaMalloc((void**)&d_state, 1 * sizeof(curandState)));
    curand_init<<<1, 1>>>(d_state, 42);

    for (int i = 0; i < n_hidden_layers; i++) {
        int n_rows = i == 0 ? N_FEATURES : layers[i - 1];
        int n_cols = i == n_hidden_layers ? N_LABELS : layers[i];
        float range = 1/sqrt(n_cols);
        int n_values = n_rows * n_cols;

        int block_size = TILE_SIZE*TILE_SIZE;
        int grid_size = (n_values + block_size - 1) / block_size;

        d_initializeWeights<<<grid_size, block_size>>>(d_weights[i], n_values, range, d_state);
    }
    
    // Print d_weights
    printf("d_weights:\n");
    for (int i = 0; i < n_hidden_layers; i++) {
        int n_rows = i == 0 ? N_FEATURES : layers[i - 1];
        int n_cols = i == n_hidden_layers ? N_LABELS : layers[i];
        int n_values = n_rows * n_cols;

        float* h_weights = (float*)malloc(n_values * sizeof(float));
        cudaMemcpy(h_weights, d_weights[i], n_values * sizeof(float), cudaMemcpyDeviceToHost);

        printf("Layer %d:\n", i);
        for (int j = 0; j < n_rows; j++) {
            for (int k = 0; k < n_cols; k++) {
                printf("%.2f ", h_weights[j * n_cols + k]);
            }
            printf("\n");
        }
        printf("\n");

        free(h_weights);
    }
    

    gpu_print_features<<<1, 32>>>(d_features);

    cudaFree(d_features);
    cuda_hello<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
    }
