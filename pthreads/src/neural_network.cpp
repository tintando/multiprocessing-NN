#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "../include/data_loading.h"
#include <string.h>
#include <pthread.h>

#define output_layer 1
#define previous_layer current_layer-1
#define next_layer current_layer+1
#define N_FEATURES 8
#define N_LABELS 1
#define NUM_THREADS 16

pthread_mutex_t mutex;// To ensure exclusive access during accomulation of biases and weights in the fast forward

//pointers for activation functions: Relu, sigmoid, tahn
typedef double (*ActivationFunction)(double);
//pointers for derivative of activation functions: dRelu, dsigmoid, dtahn
typedef double (*ActivationFunctionDerivative)(double);

/*structure of MLP
- int num_layers; // numebr of hidden layers in neural network
- int *layers_sizes; // array where each entry is the # of nodes in that layer
    ex: layer1 = 3 nodes, layer2 = 6, layer3 = 10 then hidden_layer_size=[3,6,10] and num_hidden_layers=3
- neuron_activation[layer][node] = array of pointers(without comprehending input layer) to doubles
- double **weights[layer][hideen_layers_size[layer]*hideen_layers_size[layer-1]] = (array of pointers (layer) (does not comprehend input llayer) to linearized 2D matrix)
               = weight between a node of the current layer and a node of previous layer
- double **biases array of pointers to array of biases [layer][node] does not comprehend input llayer)*/
typedef struct MLP {
    int num_layers; // numebr of hidden layers in neural network
    int *layers_sizes; // array where each entry is the # of nodes in that layer
    double **neuron_activations;//neuron activations of each layer

    // weight between a node of the current layer and a node of previous layer, (it start from first hidden layer)
    // [layer][hideen_layers_size[layer]*hideen_layers_size[layer-1]]
    double **weights; //note input lauer doesnt have weights
    double **biases;// note: input layer doesn0t have bias
} MLP;


typedef struct Data {
    double** samples; 
    double** targets;
    int size; // number of samples
}Data;

typedef struct Dataset {
    Data train;
    Data test;
    Data validation;
}Dataset;



void printData(Data data) {
    printf("Samples:\n");
    for (int i = 0; i < data.size; i++) {
        printf("Sample %d: ", i + 1);
        for (int j = 0; j < N_FEATURES; j++) {
            printf("%f ", data.samples[i][j]);
        }
        printf("\n");
    }

    printf("Targets:\n");
    for (int i = 0; i < data.size; i++) {
        printf("Target %d: ", i + 1);
        for (int j = 0; j < N_LABELS; j++) {
            printf("%f ", data.targets[i][j]);
        }
        printf("\n");
    }
}


void printMLP(const MLP *mlp) {
    printf("MLP Structure:\n");
    printf("Number of Layers: %d\n", mlp->num_layers);

    // Print sizes of each layer
    for (int i = 0; i < mlp->num_layers; i++) {
        printf("Layer %d size: %d\n", i, mlp->layers_sizes[i]);
    }

    // Print neuron activations
    for (int i = 0; i < mlp->num_layers; i++) {
        printf("Layer %d activations: ", i);
        for (int j = 0; j < mlp->layers_sizes[i]; j++) {
            printf("%lf ", mlp->neuron_activations[i][j]);
        }
        printf("\n");
    }

    // Print weights
    for (int i = 1; i < mlp->num_layers; i++) { // Start from 1 since weights are between layers
        printf("Weights to Layer %d: \n", i);
        for (int j = 0; j < mlp->layers_sizes[i]; j++) {
            for (int k = 0; k < mlp->layers_sizes[i-1]; k++) {
                printf("W[%d][%d]: %lf ", j, k, mlp->weights[i][j * mlp->layers_sizes[i-1] + k]);
            }
            printf("\n");
        }
    }

    // Print biases
    for (int i = 1; i < mlp->num_layers; i++) {
        printf("Layer %d biases: ", i);
        for (int j = 0; j < mlp->layers_sizes[i]; j++) {
            printf("%lf ", mlp->biases[i][j]);
        }
        printf("\n");
    }
}

void freeDataset(Dataset* dataset) {
    // Free train samples and targets
    for (int i = 0; i < dataset->train.size; i++) {
        free(dataset->train.samples[i]);
        free(dataset->train.targets[i]);
    }
    free(dataset->train.samples);
    free(dataset->train.targets);

    // Free test samples and targets
    for (int i = 0; i < dataset->test.size; i++) {
        free(dataset->test.samples[i]);
        free(dataset->test.targets[i]);
    }
    free(dataset->test.samples);
    free(dataset->test.targets);

    // If you have validation data, free it similarly here
}

void freeMLP(MLP *mlp) {
    if (!mlp) return; // Check if mlp is NULL

    // Free neuron activations
    if (mlp->neuron_activations) {
        for (int i = 0; i < mlp->num_layers; i++) {
            if (mlp->neuron_activations[i]) {
                free(mlp->neuron_activations[i]);
            }
        }
        free(mlp->neuron_activations);
    }

    // Free weights
    if (mlp->weights) {
        for (int i = 1; i < mlp->num_layers; i++) { // Note: weights array starts from layer 1
            if (mlp->weights[i-1]) { // Adjusted index because weights start from layer 1
                free(mlp->weights[i-1]);
            }
        }
        free(mlp->weights);
    }

    // Free biases
    if (mlp->biases) {
        for (int i = 1; i < mlp->num_layers; i++) { // Note: biases array starts from layer 1
            if (mlp->biases[i-1]) { // Adjusted index because biases start from layer 1
                free(mlp->biases[i-1]);
            }
        }
        free(mlp->biases);
    }

    // Free layers sizes
    if (mlp->layers_sizes) {
        free(mlp->layers_sizes);
    }

    // Finally, free the MLP structure itself
    free(mlp);
}

//popular way to initialize weights
//helps in keeping the signal from the input to flow well into the deep network.
void initializeXavier(double *weights, int in, int out) {
    // in = prev_layer_size
    // out = layer size
    // weights = weights between nodes of previous and current layer
    double limit = sqrt(6.0 / (in + out));
    for (int i = 0; i < in * out; i++) {
        weights[i] = (rand() / (double)(RAND_MAX)) * 2 * limit - limit;
    }
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double dsigmoid(double x) {
    double sigmoid_x = sigmoid(x);
    return sigmoid_x * (1 - sigmoid_x);
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double drelu(double x) {
    return x > 0 ? 1 : 0;
}


double dtanh(double x) {
    double tanh_x = tanh(x);
    return 1 - tanh_x * tanh_x;
}

void applyActivationFunction(double *layer, int size, ActivationFunction activationFunc) {
    for (int i = 0; i < size; i++) {
        layer[i] = activationFunc(layer[i]);
    }
}



// Allocates memory for the MLP structures and initializes them,
// including the Xavier method for initializing weights.
MLP *createMLP(int num_layers, int *layers_size) {

    MLP *mlp = (MLP *)malloc(sizeof(MLP)); //mlp is the pointer to the MLP instance
    if (!mlp){
        printf("error allocating MLP structure");
        return NULL;//in case of memory errors, this function returns NULL
    } 
    
    mlp->num_layers = num_layers;

    // allocate the array of sizes
    mlp->layers_sizes = (int *)malloc(num_layers * sizeof(int));
    if (!mlp->layers_sizes) {
        printf("error allocating layers_size");
        freeMLP(mlp);
        return NULL;
    }

    //initialize the array of sizes
    for (int i = 0; i < num_layers; i++) {
        mlp->layers_sizes[i] = layers_size[i];
    }
    // allocate neuron_activations, weights, biases
    mlp->neuron_activations = (double **)malloc(num_layers * sizeof(double *));
    mlp->weights = (double **)malloc((num_layers - 1) * sizeof(double *));
    mlp->biases = (double **)malloc((num_layers - 1) * sizeof(double *));
    if (!mlp->neuron_activations || !mlp->weights || !mlp->biases) {
        printf("error allocating neuron_activations, weights, biases");
        freeMLP(mlp);
        return NULL;
    }

    // initialize neuron_activations, weights, biases
    for (int i = 0; i < num_layers; i++) { 

        mlp->neuron_activations[i] = (double *)calloc(layers_size[i], sizeof(double)); // allocates and initializes to 0
        
        if (i!=0){// if i = 0 we are in the input layer, which doesn't have weights or biases
            mlp->weights[i] = (double *)malloc(layers_size[i-1] * layers_size[i] * sizeof(double)); //allocates
            mlp->biases[i] = (double *)calloc(layers_size[i], sizeof(double)); // allocates and initializes to 0
            if (!mlp->neuron_activations[i] || !mlp->weights[i] || !mlp->biases[i]) {
                printf("error allocating neuron_activations[%d]", i);
                freeMLP(mlp);
                return NULL;
            }
            initializeXavier(mlp->weights[i], layers_size[i-1], layers_size[i]); //initialize weights
        }

        else {
            if (!mlp->neuron_activations[i]) {
                printf("error allocating neuron_activations[0]");
                freeMLP(mlp);
                return NULL;
            }
        }
    }

        return mlp;
    }   







void loadAndPrepareDataset(const char* filename, double ***dataset, double ***targets, int *n_samples) {
    // Read the dataset
    Sample* samples = readDataset(filename, n_samples);//returns a 1D array
    if (!samples || *n_samples <= 0) {
        printf("Error reading dataset or dataset is empty.\n");
        return;
    }

    // Allocate memory to split the array in labels and features
    float* h_features = (float*)malloc(*n_samples * N_FEATURES * sizeof(float));
    float* h_labels = (float*)malloc(*n_samples * N_LABELS * sizeof(float));
    if (!h_features || !h_labels) {
        printf("Failed to allocate memory for features and labels.\n");
        delete[] samples; // Assuming samples need to be freed here
        return;
    }

    // Copy data from samples to h_features and h_labels
    for (int i = 0; i < *n_samples; i++) {
        memcpy(h_features + i * N_FEATURES, samples[i].features, N_FEATURES * sizeof(float));
        h_labels[i] = samples[i].label;
    }

    // Assuming the responsibility to free samples is here
    // Remember to free the samples' features if they're dynamically allocated
    delete[] samples;

    // Allocate memory for dataset and targets
    *dataset = (double**) malloc(*n_samples * sizeof(double*));
    *targets = (double**) malloc(*n_samples * sizeof(double*));
    for (int i = 0; i < *n_samples; i++) {
        (*dataset)[i] = (double*) malloc(N_FEATURES * sizeof(double));
        (*targets)[i] = (double*) malloc(N_LABELS * sizeof(double));
        for (int j = 0; j < N_FEATURES; j++) {
            (*dataset)[i][j] = (double)h_features[i * N_FEATURES + j];
        }
        (*targets)[i][0] = (double)h_labels[i];
    }

    // Free temporary host memory
    free(h_features);
    free(h_labels);
}

void shuffleDataset(double ***dataset, double ***targets, int n_samples) {
    srand(time(NULL)); // Seed the random number generator with current time

    for (int i = 0; i < n_samples - 1; i++) {
        int j = i + rand() / (RAND_MAX / (n_samples - i) + 1); // Generate a random index from i to n_samples-1

        // Swap dataset[i] and dataset[j]
        double *temp_dataset = (*dataset)[i];
        (*dataset)[i] = (*dataset)[j];
        (*dataset)[j] = temp_dataset;

        // Swap targets[i] and targets[j] similarly
        double *temp_targets = (*targets)[i];
        (*targets)[i] = (*targets)[j];
        (*targets)[j] = temp_targets;
    }
}


// Splits the dataset into train, validation, and test sets
Dataset splitDataset(int n_samples, double*** dataset, double*** targets){

    int train_size = (int)(n_samples*80/100), test_size = n_samples - train_size;
    double **train_data = (double**) malloc(train_size * sizeof(double*));
    double **train_targets = (double**) malloc(train_size * sizeof(double*));
    double **test_data = (double**) malloc(test_size * sizeof(double*));
    double **test_targets = (double**) malloc(test_size * sizeof(double*));

    for (int i = 0; i < train_size; i++) {
        train_data[i] = (double*) malloc(N_FEATURES * sizeof(double));
        train_targets[i] = (double*) malloc(N_LABELS * sizeof(double));
        for (int j = 0; j < N_FEATURES; j++) {
            train_data[i][j] = (*dataset)[i][j];
        }
        train_targets[i][0] = (*targets)[i][0];
    }

    for (int i = 0; i < test_size; i++) {
        test_data[i] = (double*) malloc(N_FEATURES * sizeof(double));
        test_targets[i] = (double*) malloc(N_LABELS * sizeof(double));
        for (int j = 0; j < N_FEATURES; j++) {
            test_data[i][j] = (*dataset)[i + train_size][j];
        }
        test_targets[i][0] = (*targets)[i + train_size][0];
    }

    Dataset result;
    result.train.samples = train_data;
    result.train.targets = train_targets;
    result.train.size = train_size;
    result.test.samples = test_data;
    result.test.targets = test_targets;
    result.test.size = test_size;
    // Note: If you add validation data, initialize result.validation here

    return result;
}


