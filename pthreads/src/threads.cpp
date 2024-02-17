#include "../include/dataset.h"
#include "../include/activation_functions.h"
#include "../include/mlp.h"

typedef struct Thread_args{
    long thread_id;
    MLP* mlp;
    double **my_neuron_activations; //the neuron activations for the samples of the thread (array of pointers (layers) to array of doubles)
    double **my_delta;  //array of pointers to array of doubles
    double **my_grad_weights_accumulators; //array of pointers (layer) to array of doubles (linearized 2d matrix of grad_weights)
    double **my_grad_biases_accumulator; //array of pointers(layer) to array of doubles
    int batch_start_index;
    int batch_size;
    Data* dataset; // pointer to the dataset 
    ActivationFunction act;
    ActivationFunctionDerivative dact;
    double learning_rate;
    int num_threads;
    double my_batch_loss;
}Thread_args;

void printThreadArgs(const Thread_args* args) {
    // Since thread_id is part of the args, we use it directly in the print statements.
    //if (args->thread_id!=2) return;
    printf("[%d] Thread ID: %d\n", args->thread_id, args->thread_id);
    printf("[%d] MLP Pointer: %p\n", args->thread_id, (void*)args->mlp);
    //printMLP(args->mlp, args->thread_id); // Adjust printMLP to accept thread_id if you want to integrate it similarly.

    // Print neuron activations
    printf("[%d] Neuron Activations:\n", args->thread_id);
    for (int i = 0; i < args->mlp->num_layers; i++) {
        printf("[%d] Layer %d: ", args->thread_id, i);
        for (int j = 0; j < args->mlp->layers_sizes[i]; j++) {
            printf("%lf ", args->my_neuron_activations[i][j]);
        }
        printf("\n");
    }

    // Print delta values
    printf("[%d] Delta Values:\n", args->thread_id);
    for (int i = 1; i < args->mlp->num_layers; i++) { // Starting from 1 as input layer doesn't have delta
        printf("[%d] Layer %d: ", args->thread_id, i);
        for (int j = 0; j < args->mlp->layers_sizes[i]; j++) {
            printf("%lf ", args->my_delta[i][j]);
        }
        printf("\n");
    }

    // Print gradient accumulators for weights
    printf("\nGradient Weights Accumulators:\n");
    for (int i = 1; i < args->mlp->num_layers; i++) {
        for (int j = 0; j < args->mlp->layers_sizes[i]; j++) {
            for(int k = 0; k < args->mlp->layers_sizes[i-1]; k++){
                printf("neuron %d of layer %d to neuron %d of layer %d: %lf ",k, i-1, j, i, args->my_grad_weights_accumulators[i][j * args->mlp->layers_sizes[i-1] + k]);
                printf("\n");
            }
        }
        printf("\n");
    }
    // Print gradient accumulators for biases
    printf("Gradient Biases Accumulators:\n");
    for (int i = 1; i < args->mlp->num_layers; i++) {
        printf("Layer %d: ", i);
        for (int j = 0; j < args->mlp->layers_sizes[i]; j++) {
            printf("%lf ", args->my_grad_biases_accumulator[i][j]);
        }
        printf("\n");
    }

    printf("[%d] Batch Start Index: %d\n", args->thread_id, args->batch_start_index);
    printf("[%d] Batch Size: %d\n", args->thread_id, args->batch_size);
    printf("[%d] Dataset Pointer: %p\n", args->thread_id, (void*)args->dataset);
    printf("[%d] Learning Rate: %lf\n", args->thread_id, args->learning_rate);
    printf("[%d] Number of Threads: %d\n", args->thread_id, args->num_threads);
    printf("[%d] Loss: %lf\n", args->thread_id, args->my_batch_loss);

    // Assuming act and dact are function pointers or similar identifiers
    printf("[%d] Activation Function: %p\n", args->thread_id, (void*)args->act);
    printf("[%d] Derivative of Activation Function: %p\n", args->thread_id, (void*)args->dact);
}

void *thread_hello(void *voidArgs) {
    Thread_args *args = (Thread_args *)voidArgs;
    // Use args->thread_id to print the ID you've assigned in your structure
    printf("Hello world from thread %ld\n", args->thread_id);
    return NULL;
}

void freeThreadArgs(Thread_args* args){
    for (int i = 0; i <= args->mlp->num_layers; i++) {
        free(args->my_neuron_activations[i]);
        free(args->my_delta[i]);
    }
    free(args->my_neuron_activations);
    for (int i = 0; i <= args->mlp->num_layers; i++) {
        free(args->my_grad_weights_accumulators[i]);
        free(args->my_grad_weights_accumulators[i]);
        free(args->my_grad_biases_accumulator[i]);
    }
    free(args->my_grad_weights_accumulators);
    free(args->my_grad_biases_accumulator);
    free(args);
}

/*  Allocates memory for the thread arguments and initializes them.
    Thease are the variables each main thread has
    Initializes the variables that are persistent in the epochs or while iterating the samples
    - Thread_id
    - MLP
    - Neuron activations -> the neuron activations for the samples of the thread (array of pointers (layers) to array of doubles)
    - deltas -> array of pointers to array of doubles
    - grad_weights_accumulator -> array of pointers to linearized 2d matrices 
    - grad_biases_accumulator -> array of pointers to array of doubles
    - my loss -> the loss over this batch for the thread
    */
Thread_args* createThreadArgs(MLP *mlp, long thread_id){
    //args for the specific thread
    Thread_args* args = (Thread_args*) malloc(sizeof(Thread_args));
    if (!args) return NULL;
    //printf("initializing args for thread %d\n", thread_id);
    args->thread_id = thread_id;
    //printf("thread id is %ld but it is supposed to be %ld\n", thread_id, args->thread_id);
    args->mlp = mlp;

    //initializing the array of pointers
    args->my_neuron_activations = (double**)malloc((mlp->num_layers) * sizeof(double *));
    args->my_delta = (double **)malloc((mlp->num_layers) * sizeof(double *));
    args->my_grad_weights_accumulators = (double**)malloc((mlp->num_layers) * sizeof(double *));
    args->my_grad_biases_accumulator = (double **)malloc((mlp->num_layers) * sizeof(double *));
    
    args->my_batch_loss = 0;
    args->act = sigmoid;
    args->dact = dsigmoid;

    //loop trough hidden layers and initialize persistent data structures
    for(int i = 0; i < mlp->num_layers; i++){
        
        //initialize and allocate neuron activations for the current layer
        //each layer has activations
        args->my_neuron_activations[i] = (double *)calloc(mlp->layers_sizes[i], sizeof(double));
        //initialize and allocate the gradient weights accumulator for the current layer
        //printf("layer %d has %d neurons\n", i, mlp->layers_sizes[i]);
        if (i!=0){
            //printf("the weights between layer %d and layer %d are %d\n", i, i-1, mlp->layers_sizes[i] * mlp->layers_sizes[i-1]);
            args->my_grad_weights_accumulators[i] = (double *)calloc(mlp->layers_sizes[i] * mlp->layers_sizes[i-1], sizeof(double));
        }
        args->my_grad_biases_accumulator[i] = (double *)calloc(mlp->layers_sizes[i], sizeof(double));
        args->my_delta[i] = (double *)malloc(mlp->layers_sizes[i] * sizeof(double));
    }

    // printf("\nGradient Weights Accumulators:\n");
    // for (int i = 1; i < args->mlp->num_layers; i++) {
    //     for (int j = 0; j < args->mlp->layers_sizes[i]; j++) {
    //         for(int k = 0; k < args->mlp->layers_sizes[i-1]; k++){
    //             printf("\nneuron %d of layer %d to neuron %d of layer %d: %lf ",k, i-1, j, i, args->my_grad_weights_accumulators[i][j * args->mlp->layers_sizes[i-1] + k]);
    //         }
    //     }
    // }
    // // Print gradient accumulators for biases
    // printf("\nGradient Biases Accumulators:\n");
    // for (int i = 1; i < args->mlp->num_layers; i++) {
    //     printf("Layer %d: ", i);
    //     for (int j = 0; j < args->mlp->layers_sizes[i]; j++) {
    //         printf("%lf ", args->my_grad_biases_accumulator[i][j]);
    //     }
    //     printf("\n");
    // }

    //free memory if there are errors in allocations
    if (args->my_neuron_activations == NULL || args->my_delta == NULL || args->my_grad_weights_accumulators == NULL || args->my_grad_biases_accumulator == NULL){
        freeThreadArgs(args);
        return NULL;
    }


    return args;
}

