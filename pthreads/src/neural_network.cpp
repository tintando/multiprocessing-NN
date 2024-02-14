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
input size = # of nodes in first layer
output size = # of nodes in last layer
num_hidden_layers = # of hidden layers in neural network
hideen_layers_size = array where each entry is the number of nodes in that layer
ex: layer1 = 3 nodes, layer2 = 6, layer3 = 10 then hidden_layer_size=[3,6,10] and num_hidden_layers=3
neuron_activation[layer][node] = ...
weights[layer][hideen_layers_size[layer]*hideen_layers_size[layer-1]] =
               = weight between a node of the current layer and a node of previous layer
2d array biases[layer][node]*/
typedef struct MLP {
    int input_size; // number of nodes in first layer
    int output_size; // numebr of nodes in last layer
    int num_hidden_layers; // numebr of hidden layers in neural network
    int *hidden_layers_size; // array where each entry is the # of nodes in that layer
    double **neuron_activations;
    // weight between a node of the current layer and a node of previous layer 
    // [layer][hideen_layers_size[layer]*hideen_layers_size[layer-1]]
    double **weights; 
    double **biases;
} MLP;

typedef struct Data {
    double** samples; 
    double** targets;
    int size; // number of samples
}Data;

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


typedef struct Dataset {
    Data train;
    Data test;
    Data validation;
}Dataset;

//initializing all the functions for the compiler :)

MLP *createMLP(int input_size, int output_size, int num_hidden_layers, int *hidden_layers_size);
void initializeXavier(double *weights, int in, int out);
void feedforward(MLP *mlp, double** neuron_activation, double *input, ActivationFunction act);
double backpropagation(MLP *mlp, double **inputs, double **targets, int current_batch_size, ActivationFunction act, ActivationFunctionDerivative dact, double learning_rate);
void trainMLP(MLP *mlp, double **dataset, double **targets, int num_samples, int num_epochs, double learning_rate, int batch_size, ActivationFunction act, ActivationFunctionDerivative dact);
double evaluateMLP(MLP *mlp, double **test_data, double **test_targets, int test_size, ActivationFunction act);
double sigmoid(double x);
double dsigmoid(double x);
double relu(double x);
double drelu(double x);
//double tanh(double x) already in math.h
double dtanh(double x);
void matrixMultiplyAndAddBias(double *output, double *input, double *weights, double *biases, int inputSize, int outputSize);
void applyActivationFunction(double *layer, int size, ActivationFunction activationFunc);
void initializeXavier(double *weights, int in, int out);
void loadAndPrepareDataset(const char* filename, double ***dataset, double ***targets, int *n_samples);
void shuffleDataset(double ***dataset, double ***targets, int n_samples);
void splitDataset(int n_samples,
                double*** train_data, double*** train_targets, 
                double*** test_data, double*** test_targets, 
                double*** validation_data, double*** validation_targets, 
                double*** dataset, double*** targets);
void *thread_action(void *args_p);
void *feedforward_validation(void *args_p);
void *thread_action(void *args_p);
void *feedforward_validation(void *args_p);


// Allocates memory for the MLP structures and initializes them,
// including the Xavier method for initializing weights.
MLP *createMLP(int input_size, int output_size, int num_hidden_layers, int *hidden_layers_size) {
    MLP *mlp = (MLP *)malloc(sizeof(MLP)); //mlp is the pointer to the MLP instance
    if (!mlp) return NULL;//in case of memory errors, this function returns NULL
    mlp->input_size = input_size;
    mlp->output_size = output_size;
    mlp->num_hidden_layers = num_hidden_layers;
    // allocate the array of sizes
    mlp->hidden_layers_size = (int *)malloc(num_hidden_layers * sizeof(int));
    if (!mlp->hidden_layers_size) {
        free(mlp);
        return NULL;
    }
    //initialize the array of sizes
    for (int i = 0; i < num_hidden_layers; i++) {
        mlp->hidden_layers_size[i] = hidden_layers_size[i];
    }
    // allocate neuron_activations, weights, biases
    mlp->neuron_activations = (double **)malloc((num_hidden_layers + 1) * sizeof(double *));
    mlp->weights = (double **)malloc((num_hidden_layers + 1) * sizeof(double *));
    mlp->biases = (double **)malloc((num_hidden_layers + 1) * sizeof(double *));
    if (!mlp->neuron_activations || !mlp->weights || !mlp->biases) {
        //errors
        free(mlp->hidden_layers_size);
        if (mlp->neuron_activations) free(mlp->neuron_activations);
        if (mlp->weights) free(mlp->weights);
        if (mlp->biases) free(mlp->biases);
        free(mlp);
        return NULL;
    }
    // initialize neuron_activations, weights, biases
    int prev_layer_size = input_size;//keeps track of the size of previous layer, starts with size of input
    for (int i = 0; i <= num_hidden_layers; i++) {
        int layer_size = (i == num_hidden_layers) ? output_size : hidden_layers_size[i]; //size of current layer
        mlp->neuron_activations[i] = (double *)calloc(layer_size, sizeof(double)); // allocates and initializes to 0
        mlp->weights[i] = (double *)malloc(prev_layer_size * layer_size * sizeof(double)); //allocates
        mlp->biases[i] = (double *)calloc(layer_size, sizeof(double)); // allocates and initializes to 0

        if (!mlp->neuron_activations[i] || !mlp->weights[i] || !mlp->biases[i]) {
            for (int j = 0; j < i; j++) {
                free(mlp->neuron_activations[j]);
                free(mlp->weights[j]);
                free(mlp->biases[j]);
            }
            free(mlp->neuron_activations);
            free(mlp->weights);
            free(mlp->biases);
            free(mlp->hidden_layers_size);
            free(mlp);
            return NULL;
        }

        initializeXavier(mlp->weights[i], prev_layer_size, layer_size); //initialize weights

        prev_layer_size = layer_size;
    }

    return mlp;
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
        free(samples); // Assuming samples need to be freed here
        return;
    }

    // Copy data from samples to h_features and h_labels
    for (int i = 0; i < *n_samples; i++) {
        memcpy(h_features + i * N_FEATURES, samples[i].features, N_FEATURES * sizeof(float));
        h_labels[i] = samples[i].label;
    }

    // Assuming the responsibility to free samples is here
    // Remember to free the samples' features if they're dynamically allocated
    free(samples);

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