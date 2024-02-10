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
    int batch_size;
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

__global__ void setup_kernel(curandState* state, unsigned long seed){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}


__global__ void d_initializeWeights(int layer, float** weights, int n_values, float range, curandState *state){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n_values){
        curandState localState = state[idx];
        float random = curand_uniform(&localState);
        // printf("L%d, Thread %d/%d: %f\n", layer, blockIdx.x, idx, random);
        // weights[layer][idx] = (random * 2 - 1) * range;
        weights[layer][idx] = (float) idx;
    }
}

__global__ void MatMul(float* A, float* B, float* C, int ARows, int ACols, int BRows,
    int BCols, int CRows, int CCols)
{
    float CValue = 0;

    int Row = blockIdx.y*TILE_SIZE + threadIdx.y;
    int Col = blockIdx.x*TILE_SIZE + threadIdx.x;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    for (int k = 0; k < (TILE_SIZE + ACols - 1)/TILE_SIZE; k++) {

         if (k*TILE_SIZE + threadIdx.x < ACols && Row < ARows)
             As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_SIZE + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_SIZE + threadIdx.y < BRows && Col < BCols)
             Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_SIZE + threadIdx.y)*BCols + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_SIZE; ++n)
             CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols)
        C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
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

float** d_allocateActivationShaped(int* layers, int n_layers, int batch_size){
    float** d_activations;
    checkCudaError(cudaMalloc((void**)&d_activations, n_layers * sizeof(float*)));
    float** activations = (float**)malloc(n_layers * sizeof(float*)); //array of pointers to activation matrices on device
    for(int i = 0; i < n_layers; i++) {
        int n_cols = layers[i];
        // printf("Activations %d: %d x %d\n", i, batch_size, n_cols);
        checkCudaError(cudaMalloc(&activations[i], batch_size * n_cols * sizeof(float)));
    }
    cudaMemcpy(d_activations, activations, n_layers * sizeof(float*), cudaMemcpyHostToDevice);
    return d_activations;
}

__global__ void forward_pass(MLP mlp, int batch_size, int start){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_layers = mlp.n_layers;
    int* layers = mlp.layers;
    float** weights = mlp.weights;
    float** biases = mlp.biases;
    float** activations = mlp.activations;
    float** logits = mlp.logits;
    float* inputs = mlp.inputs;
    float* labels = mlp.labels;
    
    // Forward pass
    // for(int i = 0; i < n_layers; i++){
    for(int i = 0; i < 1; i++){
        int n_rows = i == 0 ? N_FEATURES : layers[i - 1];
        int n_cols = i == n_layers ? N_LABELS : layers[i];
        int n_values = n_rows * n_cols;
        
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((n_cols + blockSize.x - 1) / blockSize.x, (batch_size + blockSize.y - 1) / blockSize.y);
        // Compute logits
        MatMul<<<gridSize, blockSize>>>(inputs+start*N_FEATURES*sizeof(float), weights[i], logits[i], batch_size, n_rows, n_rows, n_cols, batch_size, n_cols);
        cudaDeviceSynchronize();
        printf("logits[%d]: ", i);
        for(int j = 0; j < n_values; j++){
            printf("%f ", logits[i][j]);
        }
        printf("\n");
        
        // // Add biases
        // for(int j = 0; j < n_values; j++){
        //     logits[i][j] += biases[i][j];
        // }

        // // Apply activation function
        // for(int j = 0; j < n_values; j++){
        //     activations[i][j] = 1 / (1 + exp(-logits[i][j]));
        // }
    }

}

void train(MLP mlp, int epochs, int batch_size, int n_samples){
    printf("Training...\n");
    // for(int i = 0; i < epochs; i++){
    for(int i = 0; i < 1; i++){
        printf("Epoch %d\n", i);
        for (int start = 0; start < n_samples - batch_size; start += batch_size) {
            // printf("Batch %d\n", start);
        // Forward pass 
            forward_pass<<<1,1>>>(mlp, batch_size, start);
            cudaDeviceSynchronize();
        }
    }
}

void printWeights(MLP mlp) {
    int n_layers = mlp.n_layers;
    float** h_weights = (float**)malloc(n_layers * sizeof(float*));
    int* h_layers = (int*)malloc(n_layers * sizeof(int));

    cudaMemcpy(h_weights, mlp.weights, n_layers * sizeof(float*), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_layers, mlp.layers, n_layers * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n_layers; i++) {
        int n_rows = i == 0 ? N_FEATURES : h_layers[i - 1];
        int n_cols = h_layers[i];
        int n_values = n_rows * n_cols;

        float* h_weights_i = (float*)malloc(n_values * sizeof(float));
        if (h_weights_i == NULL) {
            printf("Memory allocation failed.\n");
            return;
        }

        cudaMemcpy(h_weights_i, h_weights[i], n_values * sizeof(float), cudaMemcpyDeviceToHost);

        printf("Weights for Layer %d: %dx%d\n", i, n_rows, n_cols);
        for (int j = 0; j < n_rows; j++) {
            for (int k = 0; k < n_cols; k++) {
                printf("%.2f ", h_weights_i[j * n_cols + k]);
            }
            printf("\n");
        }
        printf("\n");

        free(h_weights_i);
    }
    free(h_weights);
    free(h_layers);
}

void printActivationShaped(MLP mlp) {
    
    float** h_activations = (float**)malloc(mlp.n_layers * sizeof(float*));
    int* h_layers = (int*)malloc(mlp.n_layers * sizeof(int));

    cudaMemcpy(h_activations, mlp.activations, mlp.n_layers * sizeof(float*), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_layers, mlp.layers, mlp.n_layers * sizeof(int), cudaMemcpyDeviceToHost);


        //     int n_cols = layers[i];
        // // printf("Activations %d: %d x %d\n", i, batch_size, n_cols);
        // checkCudaError(cudaMalloc(&activations[i], batch_size * n_cols * sizeof(float)));
    
    for (int i = 0; i < mlp.n_layers; i++) {
        int n_rows = mlp.batch_size;
        int n_cols = h_layers[i];
        int n_values = n_rows * n_cols;

        float* h_activations_i = (float*)malloc(n_rows * n_cols * sizeof(float));
        if (h_activations_i == NULL) {
            printf("Memory allocation failed.\n");
            return;
        }

        cudaMemcpy(h_activations_i, h_activations[i], n_rows * n_cols * sizeof(float), cudaMemcpyDeviceToHost);

        printf("Activations for Layer %d: %dx%d\n", i, n_rows, n_cols);
        for (int j = 0; j < n_rows; j++) {
            for (int k = 0; k < n_cols; k++) {
                printf("%.2f ", h_activations_i[j * n_cols + k]);
            }
            printf("\n");
        }
        printf("\n");

        free(h_activations_i);
    }
    free(h_activations);
    free(h_layers);
}


__global__ void test_activation(MLP mlp){

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
    int* layers = (int*)malloc((n_hidden_layers+1) * sizeof(int));
    if(layers == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    // Convert arguments to integers and store them in the array
    for(int i = 1; i < argc-2; i++) {
        layers[i - 1] = atoi(argv[i]);
    }
    layers[n_hidden_layers] = N_LABELS; //output layer
    n_hidden_layers++; //TODO: rename for clarity

    int epochs = atoi(argv[argc-2]);
    int batch_size = atoi(argv[argc-1]);

    printf("Hidden layer sizes: ");
    for(int i = 0; i < n_hidden_layers; i++) {
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

    printf("HOST: Coalescing features and labels...\n");
    // coalesce features and labels
    float* h_features = (float*)malloc(n_samples * N_FEATURES * sizeof(float));
    float* h_labels = (float*)malloc(n_samples * N_LABELS * sizeof(float));
    for(int i = 0; i < n_samples; i++) {
        memcpy(h_features + i * N_FEATURES, samples[i].features, N_FEATURES * sizeof(float));
        h_labels[i] = samples[i].label;
    }
    
    printf("HOST: Allocating device memory...\n");
    // Allocate memory for the features and labels on the device and copy
    float* d_features;
    float* d_labels;
    checkCudaError(cudaMalloc((void**)&d_features, n_samples * N_FEATURES * sizeof(float)));
    cudaMemcpy(d_features, h_features, n_samples * N_FEATURES * sizeof(float), cudaMemcpyHostToDevice);
    checkCudaError(cudaMalloc((void**)&d_labels, n_samples * N_LABELS * sizeof(float)));
    cudaMemcpy(d_labels, h_labels, n_samples * N_LABELS * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate memory for the weights, biases, activations, logits, gradients, deltas
    printf("HOST: Allocating device memory for weights, biases, activations, logits, gradients, deltas...\n");
    // Allocate memory for the weights
    float** d_weights = d_allocateWeightShaped(layers, n_hidden_layers); //array of pointers to device memory

    // Allocate memory for the biases
    float** d_biases = d_allocateBiasShaped(layers, n_hidden_layers); //array of pointers to device memory

    // Allocate memory for the activations
    float** d_activations = d_allocateActivationShaped(layers, n_hidden_layers, batch_size); //array of pointers to device memory

    // Allocate memory for the logits
    float** d_logits = d_allocateActivationShaped(layers, n_hidden_layers, batch_size); //array of pointers to device memory

    // Allocate memory for the accumulated gradients
    float** d_gradients = d_allocateWeightShaped(layers, n_hidden_layers); //array of pointers to device memory

    // Allocate memory for the deltas
    float** d_deltas = d_allocateBiasShaped(layers, n_hidden_layers); //array of pointers to device memory

    int* d_layers;
    checkCudaError(cudaMalloc((void**)&d_layers, n_hidden_layers * sizeof(int)));
    cudaMemcpy(d_layers, layers, n_hidden_layers * sizeof(int), cudaMemcpyHostToDevice);

    mlp.n_layers = n_hidden_layers;
    mlp.batch_size = batch_size;
    mlp.layers = d_layers;
    mlp.weights = d_weights;
    mlp.biases = d_biases;
    mlp.activations = d_activations;
    mlp.logits = d_logits;
    mlp.gradients = d_gradients;
    mlp.deltas = d_deltas;
    mlp.inputs = d_features;
    mlp.labels = d_labels;


    printf("HOST: Initializing curand...\n");
    // Initialize curand
    curandState* d_state;
    checkCudaError(cudaMalloc((void**)&d_state, 1 * sizeof(curandState)));


    // Find the two largest layers
    int largest1 = 0;
    int largest2 = 0;
    for (int i = 0; i < n_hidden_layers; i++) {
        if (layers[i] > largest1) {
            largest2 = largest1;
            largest1 = layers[i];
        } else if (layers[i] > largest2) {
            largest2 = layers[i];
        }
    }
    int product = largest1 * largest2;
    

    setup_kernel<<<(product + 31)/32, 32>>>(d_state, 42);
    cudaDeviceSynchronize();

    printf("HOST: Initializing weights...\n");
    for (int i = 0; i < n_hidden_layers; i++) {
        int n_rows = i == 0 ? N_FEATURES : layers[i - 1];
        int n_cols = i == n_hidden_layers ? N_LABELS : layers[i];
        float range = 1/sqrt(n_cols);
        int n_values = n_rows * n_cols;

        int block_size = TILE_SIZE*TILE_SIZE;
        int grid_size = (n_values + block_size - 1) / block_size;
        
        // float* d_weights_i = mlp.weights[i]; //pointer to device memory, not HOST!!!!
        d_initializeWeights<<<grid_size, block_size>>>(i, mlp.weights, n_values, range, d_state);
        // cudaDeviceSynchronize();
    }


    // Print d_weights
    printWeights(mlp);
    printActivationShaped(mlp);
    test_activation<<<1, 1>>>(mlp);
    printActivationShaped(mlp);

    train(mlp, epochs, batch_size, n_samples);

    // printActivationShaped(mlp);
    // gpu_print_features<<<1, 32>>>(d_features);

    cudaFree(d_features);
    // cuda_hello<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
    }
