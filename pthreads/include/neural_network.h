// neural_network.h
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

// Global mutex for synchronization
extern pthread_mutex_t mutex;

// Activation functions and their derivatives
typedef double (*ActivationFunction)(double);
typedef double (*ActivationFunctionDerivative)(double);

/*structure of MLP
input size = # of nodes in first layer (int)
output size = # of nodes in last layer (int)
num_hidden_layers = # of hidden layers in neural network (int)
hideen_layers_size = array where each entry is the number of nodes in that layer (array of int)
ex: layer1 = 3 nodes, layer2 = 6, layer3 = 10 then hidden_layer_size=[3,6,10] and num_hidden_layers=3
neuron_activation[layer][node] = the value of each node during training
weights[layer][hideen_layers_size[layer]*hideen_layers_size[layer-1]] = (array of pointers (layer) to linearized 2D matrix)
               = weight between a node of the current layer and a node of previous layer
array of pointers to array of biases [layer][node]*/
typedef struct MLP {
    int input_size;
    int output_size;
    int num_hidden_layers;
    int *hidden_layers_size;
    double **neuron_activations;
    double **weights;
    double **biases;
} MLP;

// Data structure
typedef struct Data {
    double** samples;
    double** targets;
    int size; // Number of samples
} Data;

typedef struct Dataset {
    Data train;
    Data test;
    Data validation;
};

// Data handling functions
void printData(Data data);
MLP *createMLP(int input_size, int output_size, int num_hidden_layers, int *hidden_layers_size);
void initializeXavier(double *weights, int in, int out);
void loadAndPrepareDataset(const char* filename, double ***dataset, double ***targets, int *n_samples);
void shuffleDataset(double ***dataset, double ***targets, int n_samples);
Dataset splitDataset(int n_samples, double*** dataset, double*** targets);

// Activation functions
double sigmoid(double x);
double dsigmoid(double x);
double relu(double x);
double drelu(double x);
double dtanh(double x);

// Utility function for applying an activation function to a layer
void applyActivationFunction(double *layer, int size, ActivationFunction activationFunc);

#endif // NEURAL_NETWORK_H
