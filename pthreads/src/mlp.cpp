#include <stdio.h>
#include <math.h>

#define output_layer 1
#define previous_layer current_layer-1
#define next_layer current_layer+1
#define N_FEATURES 8
#define N_LABELS 1
#define NUM_THREADS 16

pthread_mutex_t mutex;// To ensure exclusive access during accomulation of biases and weights in the fast forward



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
    mlp->weights = (double **)malloc((num_layers) * sizeof(double *));
    mlp->biases = (double **)malloc((num_layers) * sizeof(double *));
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










