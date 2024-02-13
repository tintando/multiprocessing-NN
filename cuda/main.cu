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
#define BLOCK_ROWS 8


struct MLP {

    //the number of layers without considering the input
    int n_layers;

    //the batch size... what did you expect?
    int batch_size;

    //array: entry[i] = size of hidden layer i || size of output layer
    int* layers; 
    
    // array of pointers to array of floats in the device
    // the array of floats is stored so that to access weight between neuron i of previous layer
    // and neuron j of current layer you must do weights[layer][i * n_cols + j]
    // can be seen as an array if 2d matrix: weights[layer][i][j]
    float** weights; 

    // Pointer to an array of float pointers. Each float pointer within this array points
    // to an array of floats representing the biases for a single layer in the neural network.
    // Essentially, d_biases is an array where each entry corresponds to a layer in the network, and each
    // entry points to the biases for that layer stored in device memory.
    float** biases; 

    // Array of float pointers, where each float pointer points to an array of floats that represents
    // the activation values for all neurons in a single layer for a given batch of inputs.
    // Essentially, d_activations is an array where each entry corresponds to a layer in the network, and each
    // entry poins to the activation values for that layer stored in device memory.
    float** activations; 

    // Pointer to an array of float pointers. Each float pointer within this array points
    // to an array of floats that represents the logits (the inputs to the final softmax activation function
    // in classification tasks) for all neurons in a single layer for a given batch of inputs. Essentially,
    // d_logits is an array where each entry corresponds to a layer in the network, and each entry points
    // to the logits for that layer stored in device memory.
    float** logits; 

    // Declaring a pointer to an array of float pointers. Each float pointer within this array points
    // to an array of floats representing the gradients for each weight between neurons in consecutive layers
    // of the neural network. Essentially, d_gradients is an array where each entry corresponds to a layer in the
    // network, and each entry points to the gradients for that layer's weights stored in device memory.
    // This data structure is crucial for the backpropagation process where gradients are used to update weights.
    float** gradients;

    // Declaring a pointer to an array of float pointers. Each float pointer within this array points
    // to an array of floats representing the delta values for the neurons in a single layer. Delta values
    // are used in the backpropagation process to compute gradients for both weights and biases. Essentially,
    // d_deltas is an array where each entry corresponds to a layer in the network, and each entry points
    // to the delta values for that layer stored in device memory.
    float** deltas; 
    // data
    float* inputs; //array of floats 

    float* labels; //array of floats
};

// in case of a error, it will print:
//  -the file where the error was,
//  -the line that generated the error
//  -the line of code itself
//  -the error
void checkCudaError(cudaError_t error){
    if (error != cudaSuccess) {
        printf("Error: %s:%d, ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}

//creates an array of pointers(layers) that point to array of floats in the device
//floats are stored so that to access weights[layer][i][j] you must do weights[layer][i * n_cols + j]
float** d_allocateWeightShaped(int* layers, int n_layers){
    // array of pointers to array of floats in the device in the device
    // the array of floats is stored so that to access weight between neuron i of previous layer
    // and neuron j of current layer you must do weights[layer][i * n_cols + j]
    //can be seen as an array if 2d matrix: weights[layer][i][j]
    float** d_weights;

    //... allocating d_weights ...
    //  d_weights[number of hidden layers + output layer]
    checkCudaError(cudaMalloc((void**)&d_weights, (n_layers + 1) * sizeof(float*)));

    // tmp array of 2d matrix of weights.... it's gonna be sent to the device
    float** weights = (float**)malloc((n_layers + 1) * sizeof(float*));

    for(int i = 0; i < n_layers + 1; i++) {
        int n_rows = i == 0 ? N_FEATURES : layers[i - 1];// 
        int n_cols = i == n_layers ? N_LABELS : layers[i];// amount of neurons in this layer

        // printf("Weights %d: %d x %d\n", i, n_rows, n_cols);

        // allocating the 2d matrix of weight of layer i in the device
        // storing it's pointer in the host at weights[i]:
        checkCudaError(cudaMalloc(&weights[i], n_rows * n_cols * sizeof(float)));
        if(weights[i] == NULL) {
            printf("Memory allocation failed.\n");
            return NULL;
        }
    }
    //sending weights[] to the device
    cudaMemcpy(d_weights, weights, (n_layers + 1) * sizeof(float*), cudaMemcpyHostToDevice);
    return d_weights;

    //now the device has memory allocated for storing d_weights
}


// This function creates an array of pointers (layers) that point to arrays of floats (biases) in the device memory. 
// It is designed for allocating memory space for the biases of neurons in each layer of a neural network.
float** d_allocateBiasShaped(int* layers, int n_layers){
    // This line declares a pointer to an array of float pointers. Each float pointer within this array will
    // point to an array of floats representing the biases for a single layer in the neural network.
    // Essentially, d_biases is an array where each entry corresponds to a layer in the network, and each
    // entry points to the biases for that layer stored in device memory.
    float** d_biases;

    // Allocates memory in the device for the array of pointers (d_biases). The size of the allocation
    // is determined by the number of layers (n_layers), with each entry being a pointer to a float.
    // This step prepares the device memory to hold pointers to the biases arrays for each layer.
    checkCudaError(cudaMalloc((void**)&d_biases, n_layers * sizeof(float*)));

    // Allocates host memory for a temporary array of pointers (biases), similar to d_biases but in host memory.
    // This temporary array will be used to allocate and store the addresses of biases arrays before copying
    // these addresses to the device memory.
    float** biases = (float**)malloc(n_layers * sizeof(float*));

    // Loops through each layer to allocate memory for the biases of that layer.
    for(int i = 0; i < n_layers; i++) {
        // Determines the number of neurons (columns) in the current layer, which will be the size of the bias array.
        int n_cols = layers[i];
        
        // Allocates device memory for the biases of the current layer and stores the address of this memory
        // in the biases array in host memory. Each entry in biases is thus a pointer to a device memory location
        // holding the biases for a layer.
        checkCudaError(cudaMalloc(&biases[i], n_cols * sizeof(float)));
        cudaMemset(biases[i], 0, n_cols * sizeof(float));
    }

    // Copies the addresses stored in the biases array in host memory to the d_biases array in device memory.
    // After this operation, d_biases in the device memory points to the actual biases arrays for each layer,
    // also stored in device memory.
    cudaMemcpy(d_biases, biases, n_layers * sizeof(float*), cudaMemcpyHostToDevice);

    // Returns a pointer to the array of pointers in device memory. Each pointer in this array points to
    // a device memory location holding the biases for a layer. This allows the biases to be accessed
    // efficiently during computations on the device.
    return d_biases;
}


// This function creates an array of pointers (layers) that point to arrays of floats (activations) in the device memory.
// It is designed for allocating memory space for the activation values of neurons in each layer of a neural network,
// considering a specific batch size. This is essential for parallel processing of multiple inputs through the network.
float** d_allocateActivationShaped(int* layers, int n_layers, int batch_size){
    // Declares a pointer to an array of float pointers, where each float pointer will point to an array of floats
    // representing the activation values for all neurons in a single layer for a given batch of inputs.
    // Essentially, d_activations is an array where each entry corresponds to a layer in the network, and each
    // entry points to the activation values for that layer stored in device memory.
    float** d_activations;

    // Allocates memory in the device for the array of pointers (d_activations). The size of the allocation is
    // determined by the number of layers (n_layers), with each entry being a pointer to a float. This prepares
    // the device memory to hold pointers to the activation arrays for each layer.
    checkCudaError(cudaMalloc((void**)&d_activations, n_layers * sizeof(float*)));

    // Allocates host memory for a temporary array of pointers (activations), similar to d_activations but in host memory.
    // This temporary array is used to allocate and store the addresses of activation arrays before copying these
    // addresses to the device memory. It facilitates the setup of activation matrices for each layer.
    float** activations = (float**)malloc(n_layers * sizeof(float*)); // Array of pointers to activation matrices on device

    // Loops through each layer to allocate memory for the activation matrices of that layer, considering the batch size.
    for(int i = 0; i < n_layers; i++) {
        int n_cols = layers[i]; // The number of neurons in the current layer, determining the column size of the activation matrix.
        
        // Allocates device memory for the activation matrix of the current layer, considering the batch size,
        // and stores the address of this memory in the activations array in host memory. Each entry in activations
        // is thus a pointer to a device memory location holding the activation matrix for a layer. The memory is
        // also initialized to zero, preparing it for new computations.
        checkCudaError(cudaMalloc(&activations[i], batch_size * n_cols * sizeof(float)));
        // cudaMemset(activations[i], 0, batch_size * n_cols * sizeof(float)); // Initializes the memory to zero.
    }

    // Copies the addresses stored in the activations array in host memory to the d_activations array in device memory.
    // After this operation, d_activations in the device memory points to the actual activation matrices for each layer,
    // also stored in device memory. This setup allows for efficient access and manipulation of activation values during
    // the forward pass and backpropagation in the neural network.
    cudaMemcpy(d_activations, activations, n_layers * sizeof(float*), cudaMemcpyHostToDevice);

    // Returns a pointer to the array of pointers in device memory. Each pointer in this array points to a device
    // memory location holding the activation matrix for a layer, facilitating efficient computations on the device.
    return d_activations;
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

//initializes random states, one per thread
__global__ void setup_kernel(curandState* state, unsigned long seed){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &state[idx]);
}

// Initializes the array of pointers to linearized matrices 
//Note: we are in the device
__global__ void d_initializeWeights(int layer, float** weights, int n_values, float range, curandState *state){
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n_values){
        curandState localState = state[idx];
        float random = curand_uniform(&localState);
        // printf("L%d, Thread %d/%d: %f\n", layer, blockIdx.x, idx, random);
        weights[layer][idx] = (random * 2 - 1) * range;
        // weights[layer][idx] = (float) idx;
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

__global__ void logits_add_biases_activation_sigmoid(MLP mlp, int i, int n_cols){
    float* logits = mlp.logits[i];
    float* biases = mlp.biases[i];
    float* activations = mlp.activations[i];
    int n_rows = mlp.batch_size;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_values = n_rows * n_cols;
    // int col = idx % n_cols;
    // int row = idx / n_cols;
    if(idx < n_values){
        // printf("adding bias[%d] to logit[%d][%d]\n", col, row, col);
        // double check if modulo or div
        logits[idx] += biases[idx%n_cols]; //adds bias of the row
        // 1 2 3 4
        // 1 2 3 5
        // 1 2 3 6
        // 1 2 3 7

        if (i != mlp.n_layers - 1) {
            //Sigmoid
            activations[idx] = 1 / (1 + exp(-logits[idx]));
        } else {
            activations[idx] = (logits[idx] > 0) ? (logits[idx]) : (0);
        }
    }
}

__global__ void forward_pass(const MLP mlp, int batch_size, int start){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_layers = mlp.n_layers;
    int* layers = mlp.layers;
    float** weights = mlp.weights;
    float** biases = mlp.biases;
    float** activations = mlp.activations;
    float** logits = mlp.logits;
    float* inputs = mlp.inputs;
    float* labels = mlp.labels;
    // printf(" ", idx);
    // Forward pass

    for(int i = 0; i < n_layers; i++){
    // for(int i = 0; i < 1; i++){
        int n_rows = i == 0 ? N_FEATURES : layers[i - 1];
        int n_cols = i == n_layers ? N_LABELS : layers[i];
        
        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((n_cols + blockSize.x - 1) / blockSize.x, (batch_size + blockSize.y - 1) / blockSize.y);
        // Compute logits
        // printf("Computing logits for layer %d\n", i);

                // Print weights[i]
        printf("LAYER %d\n", i);
        printf("weights[%d]: %dx%d\n", i, n_rows, n_cols);
        for(int j = 0; j < n_rows; j++){
            for(int k = 0; k < n_cols; k++){
                printf("%f ", weights[i][j * n_cols + k]);
            }
            printf("\n");
        }
        printf("\n");

        // Print inputs
        printf("inputs: %dx%d\n", batch_size, n_rows);
        for(int j = 0; j < batch_size; j++){
            for(int k = 0; k < n_rows; k++){
                if (i == 0) {
                    printf("%f ", inputs[(start + j) * N_FEATURES + k]);
                } else {
                    printf("%f ", activations[i-1][j * n_rows + k]);
                }
                // printf("%f ", inputs[(start + j) * N_FEATURES + k]);
            }
            printf("\n");
        }
        printf("\n");

        if (i == 0) { // first layer
            MatMul<<<gridSize, blockSize>>>(inputs+start*N_FEATURES*sizeof(float), weights[i], logits[i], batch_size, n_rows, n_rows, n_cols, batch_size, n_cols);
        }
        else {
            MatMul<<<gridSize, blockSize>>>(activations[i-1], weights[i], logits[i], batch_size, n_rows, n_rows, n_cols, batch_size, n_cols);
        }
        cudaDeviceSynchronize();
        printf("logits[%d]:\n ", i);
        for(int j = 0; j < batch_size; j++){
            for(int k = 0; k < n_cols; k++){
                printf("%f ", logits[i][j * n_cols + k]);
            }
            printf("\n ");
        }
        printf("\n");
        
        // Add biases to logits and compute activations
        int n_values = batch_size * n_cols;
        int threads_per_block = 256;
        int blocks_per_grid = (n_values + threads_per_block - 1) / threads_per_block;
        logits_add_biases_activation_sigmoid<<<blocks_per_grid, threads_per_block>>>(mlp, i, n_cols);
        cudaDeviceSynchronize();
        
    }

}
__global__ void compute_deltas(MLP mlp, int start){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_layers = mlp.n_layers;
    int batch_size = mlp.batch_size;
    int* layers = mlp.layers;
    float** weights = mlp.weights;
    float** activations = mlp.activations;
    float** logits = mlp.logits;
    float** deltas = mlp.deltas;
    float* inputs = mlp.inputs;
    float* labels = mlp.labels + start * N_LABELS * sizeof(float);

    // Compute deltas for the output layer
    int n_rows = batch_size;
    int n_cols = layers[n_layers - 1];
    int n_values = n_rows * n_cols;

    if (idx < n_values){
        int col = idx % n_cols;
        int row = idx / n_cols;
        
        // (predicted-target) [hadamard] step(logits)
        deltas[n_layers - 1][row * n_cols + col] = (activations[n_layers-1][row * n_cols + col] - labels[row * n_cols + col]) * (activations[n_layers - 1][row * n_cols + col] >= 0) ? (1) : (0); 
        // deltas[n_layers - 1][row * n_cols + col] = (float)idx;


        // Compute deltas for the hidden layers
        for(int i = n_layers - 2; i > 0; i--){
            // int n_rows = layers[i - 1];
            int n_cols = layers[i];
            int n_values = n_rows * n_cols;
            float sum = 0;
            if (idx < n_values){
                //d^(l+1)*W^l+1^T [hadamard] afunc'(logit^l)
                //BSIZExLAYERS[i+1] LAYERS[i+1]xLAYERS[i] = BSIZExLAYERS[i] [hadamard] sig(logit^l)(1-sig(logit^l))
                //for every logits compute sig*1-sig, for value in delta multiply by corresponding logit
                //resulting delta contains on each row the deltas for a sample, and columns the neurons
                if (idx == 0){
                    // matmul
                    dim3 blockSize(TILE_SIZE, TILE_SIZE);
                    dim3 gridSize((n_cols + blockSize.x - 1) / blockSize.x, (batch_size + blockSize.y - 1) / blockSize.y);
                    MatMul<<<gridSize, blockSize>>>(deltas[i+1], weights[i+1], deltas[i], batch_size, layers[i+1], layers[i+1], layers[i], batch_size, layers[i]);
                    cudaDeviceSynchronize();
                }
                __syncthreads();
                deltas[i][idx] *= (1 - logits[i][idx]) * logits[i][idx];
                // each thread computes sig*1-sig and multiplies to their logit

                __syncthreads();
            }
        }
        
        if (idx == 0) {
            printf("deltas[%d]:\n", idx);
            for(int i = 0; i < n_layers; i++){
                int n_rows = batch_size;
                int n_cols = layers[i];
                for(int j = 0; j < n_rows; j++){
                    for(int k = 0; k < n_cols; k++){
                        printf("%f ", deltas[i][j * n_cols + k]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            printf("deltas[%d]: %f\n", idx, deltas[n_layers - 1][row * n_cols + col]);
            __syncthreads();
        }
    }
}


// __global__ void accumulate_bias_gradients(MLP mlp){
//     int layer = blockIdx.z;
    

//     // Bias gradients
//     for(int i = 0; i < n_layers; i++){
//         int n_cols = layers[i];
//         int n_values = n_cols;
//         if (idx < n_values){
//             for(int j = 0; j < mlp.batch_size; j++){
//                 gradients[i][idx] += deltas[i][j * n_cols + idx];
//             }
//         }
//     }
// }


__global__ void column_sum(const float* matrix, const unsigned width, const unsigned height, float* result){

    unsigned idx = threadIdx.x + blockDim.x*blockIdx.x;
    while (idx < width){
        float my_result = (float)0;
        for (int i = 0; i < height; i++) my_result += matrix[(i*width)+idx]/height;
        result[idx] = my_result;
        idx += gridDim.x * blockDim.x;}

}

__global__ void transposeCoalesced(float *odata, float *idata, int width, int height)
{
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    int xIndex = blockIdx.x * TILE_SIZE + threadIdx.x;
    int yIndex = blockIdx.y * TILE_SIZE + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;
    xIndex = blockIdx.y * TILE_SIZE + threadIdx.x;
    yIndex = blockIdx.x * TILE_SIZE + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS)
    {
        tile[threadIdx.y + i][threadIdx.x] =
            idata[index_in + i * width];
    }
    __syncthreads();
    for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS)
    {
        odata[index_out + i * height] =
            tile[threadIdx.x][threadIdx.y + i];
    }

}

__global__ void backpropagation(MLP mlp, int start){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n_layers = mlp.n_layers;
    int batch_size = mlp.batch_size;
    int* layers = mlp.layers;
    float** weights = mlp.weights;
    float** biases = mlp.biases;
    float** activations = mlp.activations;
    float** logits = mlp.logits;
    float** gradients = mlp.gradients;
    float** deltas = mlp.deltas;
    float* inputs = mlp.inputs;
    float* labels = mlp.labels;

    // Deltas
    int n_values = mlp.layers[mlp.n_layers - 2] * mlp.layers[mlp.n_layers - 1];
    int threads_per_block = 256;
    int blocks_per_grid = (n_values + threads_per_block - 1) / threads_per_block;
    compute_deltas<<<blocks_per_grid, threads_per_block>>>(mlp, start);
    // Bias gradients
    // int largestProduct = N_FEATURES * layers[0];
    // for (int i = 0; i < n_layers - 1; i++) {
    //     int product = layers[i] * layers[i+1];
    //     if (product > largestProduct) {
    //         largestProduct = product;
    //     }
    // }

    int largestItem = layers[0];
    for (int i = 1; i < n_layers; i++) {
        if (layers[i] > largestItem) {
            largestItem = layers[i];
        }
    }

    // Bias gradients
    for (int i = 0; i < n_layers; i++) {
        int width = layers[i];
        int height = batch_size;
        column_sum<<<(width*height + 255)/256, 256>>>(deltas[i], width, height, biases[i]);
    }

        // int n_rows = i == 0 ? N_FEATURES : layers[i - 1];
        // int n_cols = i == n_layers ? N_LABELS : layers[i];
        
        // dim3 blockSize(TILE_SIZE, TILE_SIZE);
        // dim3 gridSize((n_cols + blockSize.x - 1) / blockSize.x, (batch_size + blockSize.y - 1) / blockSize.y);
        // MatMul<<<gridSize, blockSize>>>(activations[i-1], weights[i], logits[i], batch_size, n_rows, n_rows, n_cols, batch_size, n_cols);

    // Weight gradients
    // I should be getting a weight shaped matrix for each row of activations :(
    for (int i = 0; i < n_layers - 1; i++) {
        // deltas * activations^-1
        // LAYER[i]xBSIZE * BSIZE*LAYER[i-1] = LAYER[i]xLAYER[i-1]
        // LAYER[i-1]xBSIZE * BSIZE*LAYER[i] = LAYER[i-1]xLAYER[i]
        // acts-1^T               deltas
        float* temp_transpose_activations;
        checkCudaError(cudaMalloc(&temp_transpose_activations, layers[i-1] * batch_size * sizeof(float)));
        dim3 gridSize((layers[i-1] + TILE_SIZE - 1) / TILE_SIZE, (batch_size + TILE_SIZE - 1) / TILE_SIZE);
        dim3 blockSize(TILE_SIZE, BLOCK_ROWS);
        
        transposeCoalesced<<<gridSize, blockSize>>>(temp_transpose_activations, activations[i-1], layers[i-1], batch_size);
        cudaDeviceSynchronize();

        int n_rows = layers[i];
        int n_cols = layers[i+1];
        int n_values = n_rows * n_cols;

        dim3 blockSize(TILE_SIZE, TILE_SIZE);
        dim3 gridSize((n_cols + blockSize.x - 1) / blockSize.x, (batch_size + blockSize.y - 1) / blockSize.y);
        MatMul<<<blocks_per_grid, threads_per_block>>>(temp_transpose_activations, deltas[i], gradients[i], batch_size, n_rows, batch_size, n_cols, n_rows, n_cols);
        
    }

    // Backward pass
    // for(int i = n_layers - 1; i >= 0; i--){
    //     int n_rows = i == 0 ? N_FEATURES : layers[i - 1];
    //     int n_cols = i == n_layers ? N_LABELS : layers[i];
    //     int n_values = n_rows * n_cols;
    //     int threads_per_block = 256;
    //     int blocks_per_grid = (n_values + threads_per_block - 1) / threads_per_block;
    //     if(i == n_layers - 1){
    //         // Compute deltas for the output layer
    //         // printf("Computing deltas for output layer\n");
    //         for(int j = 0; j < n_values; j++){
    //             deltas[i][j] = (activations[i][j] - labels[j]) * (1 - activations[i][j]) * activations[i][j];
    //         }
    //         // printf("deltas[%d]:\n ", i);
    //         // for(int j = 0; j < n_rows; j++){
    //         //     for(int k = 0; k < n_cols; k++){
    //         //         printf("%f ", deltas[i][j * n_cols + k]);
    //         //     }
    //         //     printf("\n ");
    //         // }
    //         // printf("\n");
    //     }
    //     else{
    //         // Compute deltas for the hidden layers
    //         // printf("Computing deltas for hidden layer %d\n", i);
    //         int n_cols_next = layers[i + 1];
    //         int n_values_next = n_cols_next * n_cols;
    //         int threads_per_block = 256;
    //         int blocks_per_grid = (n_values + threads_per_block - 1) / threads_per_block;
    //         for(int j = 0; j < n_values; j++){
    //             float sum = 0;
    //             for(int k = 0; k < n_values_next; k++){
    //                 sum += weights[i + 1][k] * deltas[i + 1][k];
    //             }
    //             deltas[i][j] = (1 - activations[i][j]) * activations[i][j] * sum;
    //         }
        

        
    //     }
    // }

}
void train(const MLP mlp, int epochs, int n_samples){
    printf("Training...\n");
    
    int n_layers = mlp.n_layers; //the number of layers without considering the input
    int batch_size = mlp.batch_size;// the batch size... what did you expect?
    
    //array: entry[i] = size of hidden layer i || size of output layer
    //it is taken from the device to keep modularity of the function
    int* h_layers = (int*)malloc(n_layers * sizeof(int));
    cudaMemcpy(h_layers, mlp.layers, n_layers * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < n_layers; i++) {//for each layer
        printf("Layer %d: %d\n", i, h_layers[i]);//print the size of the layer
    }


    for(int i = 0; i < epochs; i++){     // for(int i = 0; i < 1; i++){ //for each epoch
        printf("Epoch %d\n", i);

        //for each batch
        // for (int start = 0; start < n_samples - batch_size; start += batch_size) {         // for (int start = 0; start < 1; start += batch_size) {
        for (int start = 0; start < 1; start += batch_size) {
        // Forward pass 
            //started as gpu function to avoid launching memcpy
            //it will launch other threads itself
            forward_pass<<<1,1>>>(mlp, batch_size, start);
            cudaDeviceSynchronize();
            // to ensure Forward pass -> backpropagation

        // Backward pass
            backpropagation<<<1,1>>>(mlp, start);
            cudaDeviceSynchronize();
        }
    }
}

void printWeights(const MLP mlp) {
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

__global__ void print_layers(const MLP mlp){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0){
        printf("Number of layers: %d\n", mlp.n_layers);
        for(int i = 0; i < mlp.n_layers; i++){
            printf("Layer %d: %d\n", i, mlp.layers[i]);
        }
    }
}

__global__ void d_printActivationShaped(MLP mlp){
    for (int i = 0; i < mlp.n_layers; i++) {
        int n_rows = mlp.batch_size;
        int n_cols = mlp.layers[i];
        int n_values = n_rows * n_cols;

        printf("Activations for Layer %d: %dx%d\n", i, n_rows, n_cols);
        for (int j = 0; j < n_rows; j++) {
            for (int k = 0; k < n_cols; k++) {
                printf("%.2f ", mlp.activations[i][j * n_cols + k]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


void printActivationShaped(const MLP mlp, int* layers, int choice) {
    float** h_activations = (float**)malloc(mlp.n_layers * sizeof(float*)); // activation shaped, may contain logits, activations, gradients, deltas
    // int* h_layers = (int*)malloc(mlp.n_layers * sizeof(int));
    // int* h_layers = layers;
    char* parameter;
    switch (choice) {
        case 1: // Copy mlp.logits
            cudaMemcpy(h_activations, mlp.logits, mlp.n_layers * sizeof(float*), cudaMemcpyDeviceToHost);
            parameter = "Logits";
            break;
        case 2: // Copy mlp.activations
            cudaMemcpy(h_activations, mlp.activations, mlp.n_layers * sizeof(float*), cudaMemcpyDeviceToHost);
            parameter = "Activations";
            break;
        case 3: // Copy mlp.deltas
            cudaMemcpy(h_activations, mlp.deltas, mlp.n_layers * sizeof(float*), cudaMemcpyDeviceToHost);
            parameter = "Deltas";
            break;
        case 4: // Copy mlp.gradients
            cudaMemcpy(h_activations, mlp.gradients, mlp.n_layers * sizeof(float*), cudaMemcpyDeviceToHost);
            parameter = "Gradients";
            break;
        default:
            printf("Invalid choice.\n");
            break;
    }

    // cudaMemcpy(h_activations, mlp.activations, mlp.n_layers * sizeof(float*), cudaMemcpyDeviceToHost);

    // h_layers = layers;
    for (int i = 0; i < mlp.n_layers; i++) {
        int n_rows = mlp.batch_size;
        int n_cols = layers[i];
        int n_values = n_rows * n_cols;

        float* h_activations_i = (float*)malloc(n_rows * n_cols * sizeof(float));
        if (h_activations_i == NULL) {
            printf("h_activations_i Memory allocation failed.\n");
            exit(1);
        }

        cudaMemcpy(h_activations_i, h_activations[i], n_rows * n_cols * sizeof(float), cudaMemcpyDeviceToHost);

        printf("%s for Layer %d: %dx%d\n", parameter, i, n_rows, n_cols);
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
    // free(h_layers);
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
    
    int n_hidden_layers = argc - 3;// the number of hidden layers
    //array: entry[i] = size of hidden layer i || size of output layer
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
    int n_layers = ++n_hidden_layers; //number of layers without counting the input layer

    int epochs = atoi(argv[argc-2]);
    int batch_size = atoi(argv[argc-1]);

    printf("Hidden layer sizes: ");
    for(int i = 0; i < n_layers; i++) {
        printf("%d ", layers[i]);
    }
    
    printf("\nEpochs: %d\n", epochs);
    printf("Batch size: %d\n\n", batch_size);


    // Load the dataset (must have 8 features and 1 target)
    char* filename = "/home/tintando/Documents/multiprocessing-NN/cuda/datasets/california.csv";

    // Read the dataset

    int n_samples;// the number of samples
    Sample* samples = readDataset(filename, &n_samples);//array of: (features[8], label)
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
    printf("HOST: Coalescing features and labels...\n");
    
    float* h_features = (float*)malloc(n_samples * N_FEATURES * sizeof(float));//array of features (on the host)
    float* h_labels = (float*)malloc(n_samples * N_LABELS * sizeof(float));//array of labels (on the host)
    for(int i = 0; i < n_samples; i++) {
        //copy sample by sample from host to device
        //can be read as: 
        // move in h_features[i*N_FEATURES] = samples[i].features
        memcpy(h_features + i * N_FEATURES, samples[i].features, N_FEATURES * sizeof(float));
        h_labels[i] = samples[i].label;
    }
    
    printf("HOST: Allocating device memory...\n");
    
    float* d_features;// array of features (on the device)
    float* d_labels;// array of labels (on the device)

    checkCudaError(cudaMalloc((void**)&d_features, n_samples * N_FEATURES * sizeof(float))); //allocates features in device memory  
    cudaMemcpy(d_features, h_features, n_samples * N_FEATURES * sizeof(float), cudaMemcpyHostToDevice);// initializes features in device memory  
    checkCudaError(cudaMalloc((void**)&d_labels, n_samples * N_LABELS * sizeof(float)));// allocates labels in device memory  
    cudaMemcpy(d_labels, h_labels, n_samples * N_LABELS * sizeof(float), cudaMemcpyHostToDevice);// initializes features in device memory  

    // Allocate memory for the weights, biases, activations, logits, gradients, deltas
    printf("HOST: Allocating device memory for weights, biases, activations, logits, gradients, deltas...\n");

    // array of pointers to array of floats in the device
    // the array of floats is stored so that to access weight between neuron i of previous layer
    // and neuron j of current layer you must do weights[layer][i * n_cols + j]
    // can be seen as an array if 2d matrix: weights[layer][i][j]
    float** d_weights = d_allocateWeightShaped(layers, n_layers);

    // Declaring a pointer to an array of float pointers. Each float pointer within this array points
    // to an array of floats representing the biases for a single layer in the neural network.
    // Essentially, d_biases is an array where each entry corresponds to a layer in the network, and each
    // entry points to the biases for that layer stored in device memory.
    float** d_biases = d_allocateBiasShaped(layers, n_layers); 

    // array of float pointers, where each float pointer points to an array of floats that represents
    // the activation values for all neurons in a single layer for a given batch of inputs.
    // Essentially, d_activations is an array where each entry corresponds to a layer in the network, and each
    // entry poins to the activation values for that layer stored in device memory.
    float** d_activations = d_allocateActivationShaped(layers, n_layers, batch_size); 

    /*The d_allocateWeightShaped function can allocate memory for both weights and their corresponding gradients since they share the same shape
    The d_allocateBiasShaped function can be used for both since biases and deltas have a shape corresponding to the number of neurons in each layer.
    The d_allocateActivationShaped function suits both purposes because it allocates memory based on the number of neurons and the batch size, which applies to both activations and logits.*/

    // Declaring a pointer to an array of float pointers. Each float pointer within this array points
    // to an array of floats representing the gradients for each weight between neurons in consecutive layers
    // of the neural network. Essentially, d_gradients is an array where each entry corresponds to a layer in the
    // network, and each entry points to the gradients for that layer's weights stored in device memory.
    // This data structure is crucial for the backpropagation process where gradients are used to update weights.
    float** d_gradients = d_allocateWeightShaped(layers, n_layers); 

    // Declaring a pointer to an array of float pointers. Each float pointer within this array points
    // to an array of floats representing the delta values for the neurons in a single layer. Delta values
    // are used in the backpropagation process to compute gradients for both weights and biases. Essentially,
    // d_deltas is an array where each entry corresponds to a layer in the network, and each entry points
    // to the delta values for that layer stored in device memory.
    float** d_deltas = d_allocateBiasShaped(layers, n_layers); 

    // Declaring a pointer to an array of float pointers. Each float pointer within this array points
    // to an array of floats that represents the logits (the inputs to the final softmax activation function
    // in classification tasks) for all neurons in a single layer for a given batch of inputs. Essentially,
    // d_logits is an array where each entry corresponds to a layer in the network, and each entry points
    // to the logits for that layer stored in device memory.
    float** d_logits = d_allocateActivationShaped(layers, n_layers, batch_size); 

    //creating d_layer variable, which is a list of ints, the same as layers
    int* d_layers; 
    checkCudaError(cudaMalloc((void**)&d_layers, n_layers * sizeof(int)));
    cudaMemcpy(d_layers, layers, n_layers * sizeof(int), cudaMemcpyHostToDevice);

    //creating mlp structure that contains all Multi_layer_perceptron stuff
    mlp.n_layers = n_layers;
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

    // int* h_layers = (int*)malloc(mlp.n_layers * sizeof(int));
    // cudaMemcpy(h_layers, mlp.layers, mlp.n_layers * sizeof(int), cudaMemcpyDeviceToHost);


    //INITIALIZING WEIGHTS 
    
    printf("HOST: Initializing curand...\n");
    // array of curandState, one for each cuda thread (to ensure random values independence between the threads)
    curandState* d_state;

    /*(void**)&d_state:  Since cudaMalloc gives the pointer to the new allocated memory 
                         exclusively to a type**. 
                         So here it: 1)takes pointer of d_state
                                     2)casts it to void**
                         
    */
    checkCudaError(cudaMalloc((void**)&d_state, 1 * sizeof(curandState)));


    // Find the two largest adjacent layers
    // since for each layer, we initialize at most largestProduct weights
    // we'll need at most largestProduct curandState
    int largestProduct = N_FEATURES * layers[0];
    for (int i = 0; i < mlp.n_layers - 1; i++) {
        int product = layers[i] * layers[i+1];
        if (product > largestProduct) {
            largestProduct = product;
        }
    }
    

    // +31 is to round by excess the number of blocks
    //             (number of blocks, threads per block)
    setup_kernel<<<(largestProduct + 31)/32, 32>>>(d_state, 42);
    cudaDeviceSynchronize();

    printf("HOST: Initializing weights...\n");
    for (int i = 0; i < n_layers; i++) {// for each layer
        int n_rows = i == 0 ? N_FEATURES : layers[i - 1]; //previous layer
        int n_cols = i == n_layers ? N_LABELS : layers[i]; //current layer
        float range = 1/sqrt(n_cols);
        int n_values = n_rows * n_cols;// weights between the layers

        int block_size = TILE_SIZE*TILE_SIZE;
        int grid_size = (n_values + block_size - 1) / block_size;
        
        // float* d_weights_i = mlp.weights[i]; //pointer to device memory, not HOST!!!!

        //<<<grid_size, block_size>>> <= <<<(largestProduct + 31)/32, 32>>>
        //so there are enough random states for every weight
        d_initializeWeights<<<grid_size, block_size>>>(i, mlp.weights, n_values, range, d_state);
        // cudaDeviceSynchronize();
    }


    // Print d_weights
    printWeights(mlp);
    // printActivationShaped(mlp);
    // test_activation<<<1, 1>>>(mlp);
    printActivationShaped(mlp, layers, 1);

    train(mlp, epochs, n_samples);

    // d_printActivationShaped<<<1,1>>>(mlp);

    // gpu_print_features<<<1, 32>>>(d_features);

    printActivationShaped(mlp, layers, 1); // logits
    printActivationShaped(mlp, layers, 2); // activations
    // printActivationShaped(mlp, layers, 3); // deltas

    // cuda_hello<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
    }

