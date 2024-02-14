#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "include/data_loading.h"
#include <string.h>
#include <pthread.h>
#include "include/neural_network.h"

#define output_layer 1
#define previous_layer current_layer-1
#define next_layer current_layer+1
#define N_FEATURES 8
#define N_LABELS 1
#define NUM_THREADS 1


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
    double my_loss;
}Thread_args;

void printThreadArgs(const Thread_args* args) {
    printf("Thread ID: %d\n", args->thread_id);
    printf("MLP Pointer: %p\n", (void*)args->mlp);
    //printMLP(args->mlp); // Print details of the MLP structure

    // Print neuron activations
    printf("Neuron Activations:\n");
    for (int i = 0; i < args->mlp->num_layers; i++) {
        printf("Layer %d: ", i);
        for (int j = 0; j < args->mlp->layers_sizes[i]; j++) {
            printf("%lf ", args->my_neuron_activations[i][j]);
        }
        printf("\n");
    }

    // Print delta values
    printf("Delta Values:\n");
    for (int i = 1; i < args->mlp->num_layers; i++) { // Starting from 1 as input layer doesn't have delta
        printf("Layer %d: ", i);
        for (int j = 0; j < args->mlp->layers_sizes[i]; j++) {
            printf("%lf ", args->my_delta[i][j]);
        }
        printf("\n");
    }

    // Print gradient accumulators for weights
    printf("Gradient Weights Accumulators:\n");
    for (int i = 1; i < args->mlp->num_layers; i++) {
        printf("Layer %d: ", i);
        for (int j = 0; j < args->mlp->layers_sizes[i] * args->mlp->layers_sizes[i-1]; j++) {
            printf("%lf ", args->my_grad_weights_accumulators[i][j]);
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

    printf("Batch Start Index: %d\n", args->batch_start_index);
    printf("Batch Size: %d\n", args->batch_size);
    printf("Dataset Pointer: %p\n", (void*)args->dataset);
    printf("Learning Rate: %lf\n", args->learning_rate);
    printf("Number of Threads: %d\n", args->num_threads);
    printf("Loss: %lf\n", args->my_loss);

    // Assuming act and dact are function pointers or similar identifiers
    printf("Activation Function: %p\n", (void*)args->act);
    printf("Derivative of Activation Function: %p\n", (void*)args->dact);
}


void matrixMultiplyAndAddBias(double *output, double *input, 
                              double *weights, double *biases, 
                              int layers_sizes_i_minus_one, int layers_sizes_i) {
    /*  j = neuron of current layer
        k = neuron of previous layer
    */                            
   //iterate trough neurons of current layer
    for (int j = 0; j < layers_sizes_i; j++) {
        output[j] = 0.0;// initialize
        for (int k = 0; k < layers_sizes_i_minus_one; k++) {// neuron of previous layer
            //multiply each input neuron by corresponding weight
            output[j] += input[k] * weights[j * layers_sizes_i_minus_one + k]; 
        }
        output[j] += biases[j];//add bias
    }
}

void feedforward_thread(Thread_args* args){
    // gives input
   MLP *mlp = args->mlp;
    
    for (int j = 0; j < mlp->layers_sizes[0]; j++) {
        // Initialize the activation of the input layer neurons with the input values.
        args->my_neuron_activations[0][j] = args->dataset->samples[0][j];
    }

    // compute neuron activation for the hidden layers
    for (int i = 1; i < mlp->num_layers; i++) {
         // for each hidden layer
        matrixMultiplyAndAddBias(args->my_neuron_activations[i], args->my_neuron_activations[i-1], args->mlp->weights[i],
                                args->mlp->biases[i], args->mlp->layers_sizes[i-1], args->mlp->layers_sizes[i]);
        
        applyActivationFunction(args->my_neuron_activations[i], args->mlp->layers_sizes[i], args->act);
    }
}




void *thread_action(void *voidArgs){

    Thread_args *args = (Thread_args *)voidArgs;//casting the correct type to args
    printf("Hello world from thread %ld\n", args->thread_id);
    double batch_loss = 0;

    //if it is last thread, it has less samples
    int my_number_of_samples = (args->thread_id != NUM_THREADS-1) ? args->batch_size/NUM_THREADS : args->batch_size % NUM_THREADS;
    int my_start_index = args->thread_id * my_number_of_samples;
    int my_end_index = my_start_index + my_number_of_samples;

    //iterate trough my samples
    printThreadArgs(args);
    for (int sample= 0; sample<my_end_index-my_start_index+1; sample++) {
        printThreadArgs(args);
        double sample_loss = 0;
        feedforward_thread(args);

        //backpropagation_thread(args)

        // for (int i = 0; i < args->mlp->output_size; i++) {// for each output node
        //         // error = result - expected
        //         double output_error = args->dataset->samples[sample][i] - args->my_neuron_activations[args->mlp->num_hidden_layers][i];
        //         // delta = error * derivativeofactivationfunction(value_of_output_node_i)
        //         args->my_delta[args->mlp->num_hidden_layers][i] = output_error * args->dact(args->mlp->neuron_activations[args->mlp->num_hidden_layers][i]);
        //         //This step quantifies how each output neuron's activation needs to change to reduce the overall error.
        //         sample_loss+=output_error*output_error;
        // }
        // batch_loss+=sample_loss;
    }
    printThreadArgs(args);
    return NULL;
}

void *thread_hello(void *voidArgs) {
    Thread_args *args = (Thread_args *)voidArgs;
    // Use args->thread_id to print the ID you've assigned in your structure
    printf("Hello world from thread %ld\n", args->thread_id);
    return NULL;
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
    printf("initializing args for thread %d\n", thread_id);
    args->thread_id = thread_id;
    printf("thread id is %ld but it is supposed to be %ld\n", thread_id, args->thread_id);
    args->mlp = mlp;

    //initializing the array of pointers
    args->my_neuron_activations = (double**)malloc((mlp->num_layers) * sizeof(double *));
    args->my_delta = (double **)malloc((mlp->num_layers) * sizeof(double *));
    args->my_grad_weights_accumulators = (double**)malloc((mlp->num_layers) * sizeof(double *));
    args->my_grad_biases_accumulator = (double **)malloc((mlp->num_layers) * sizeof(double *));
    
    args->my_loss = 0;
    args->act = relu;
    args->dact = drelu;

    //loop trough hidden layers and initialize persistent data structures
    for(int i = 0; i <= mlp->num_layers; i++){
        
        //initialize and allocate neuron activations for the current layer
        //each layer has activations
        args->my_neuron_activations[i] = (double *)calloc(mlp->layers_sizes[i], sizeof(double));
        if (i!=0){// every layer except the input has biases, deltas and weights (deltas have same shape of biases)
            args->my_grad_weights_accumulators[i] = (double *)malloc(mlp->layers_sizes[i] * mlp->layers_sizes[i-1] * sizeof(double));
            args->my_grad_biases_accumulator[i] = (double *)calloc(mlp->layers_sizes[i], sizeof(double));
            args->my_delta[i] = (double *)malloc(mlp->layers_sizes[i] * sizeof(double));
        }
    }

    //if there were errors, free everything
    // if(!args->my_neuron_activations || !args->my_grad_weights_accumulators || !args->my_grad_biases_accumulator || !args->my_delta){
    //     printf("Problem during allocatin of thread_args");
    //     for (int i = 0; i <= mlp->num_hidden_layers; i++) {
    //         free(args->my_neuron_activations[i]);
    //         free(args->my_delta[i]);
    //     }
    //     free(args->my_neuron_activations);
    //     for (int i = 0; i <= mlp->num_hidden_layers; i++) {
    //         free(args->my_grad_weights_accumulators[i]);
    //         free(args->my_grad_weights_accumulators[i]);
    //         free(args->my_grad_biases_accumulator[i]);
    //     }
    //     free(args->my_grad_weights_accumulators);
    //     free(args->my_grad_biases_accumulator);
    //     free(args);
    //     return NULL;
    // }

    return args;
}



void trainMLP(Data train_dataset, MLP* mlp, int num_epochs, int batch_size, int learning_rate){
    
    //initialize thread data structures
    pthread_t threads[NUM_THREADS]; //thread identifier
    Thread_args* thread_args[NUM_THREADS]; // array of thread data, one specific for thread

    //Initializes the variables that are persistent in the epochs or data stractures that keep the same shape
    for(long thread_id=0; thread_id < NUM_THREADS; thread_id++){
        thread_args[thread_id] = createThreadArgs(mlp,thread_id); 
        thread_args[thread_id]->dataset = &train_dataset;
        //printThreadArgs(thread_args[thread]);
        // printf("outside the thread: %ld\n", thread_args[thread_id]->thread_id);
        // pthread_create(&threads[thread_id], NULL,  thread_hello, (void *)thread_args[thread_id]);
        // pthread_join(threads[thread_id], NULL);
    }

    //for each epoch
//for (int epoch = 0; epoch < num_epochs; epoch++) {
        //printf("epoch %d: \n", epoch);
        double epoch_loss = 0.0; //accomulator of loss over a single epoch

        // iterate through the dataset in batches
        train_dataset.size = 5; //tmp
for (int batch_start_index = 0; batch_start_index < train_dataset.size; batch_start_index += batch_size) { 
            //printData(train_dataset);
            //the size of the ith batch.
            int current_batch_size = (batch_start_index + batch_size > train_dataset.size) ? (train_dataset.size - batch_start_index) : batch_size;
            // double **batch_inputs = (double **)malloc(current_batch_size * sizeof(double *));// the inputs for this batch
            // double **batch_targets = (double **)malloc(current_batch_size * sizeof(double *));// the labels for this batch

            // for (int j = 0; j < current_batch_size; j++) {
            //         batch_inputs[j] = dataset[i + j];
            //         batch_targets[j] = targets[i + j];
            //     }


            // initializing data structures of threads that are dependent on the batch
            for (long thread_id = 0; thread_id < NUM_THREADS; ++thread_id) {
                
                thread_args[thread_id]->batch_size = current_batch_size;
                thread_args[thread_id]->batch_start_index = batch_start_index;
                //starting the threads
                printf("creating thread %d\n", thread_id);
                pthread_create(&threads[thread_id], NULL,  thread_action, (void *)thread_args[thread_id]);
            }

            for(long thread_id = 0; thread_id < NUM_THREADS; thread_id++){
                pthread_join(threads[thread_id], NULL);
            }

        }
     //}
    
 }

int main(int argc, char *argv[]){

    const char* filename = "/home/lexyo/Documenti/Dev/Multicore/multiprocessing-NN/pthreads/datasets/california.csv";
    double **dataset = NULL, **targets = NULL;
    int n_samples = 0;

    // Load and prepare the dataset
    loadAndPrepareDataset(filename, &dataset, &targets, &n_samples);
    
    Dataset splitted_dataset = splitDataset(n_samples, &dataset, &targets);

    //can be freed, we don't need them anymore
    // for (int i = 0; i < n_samples; i++) {
    //      free(dataset[i]);
    //      free(targets[i]);
    //  }
    
    // free(dataset);
    // free(targets);
    

    //printData(splitted_dataset.train);
    // Initialize your MLP
    int input_size = N_FEATURES; // Define according to your dataset
    int output_size = N_LABELS; // Typically 1 for regression tasks
    int num_layers = 4; // Example: 4 layers (2 hidden)
    int layers_size[] = {N_FEATURES, 5, 2, N_LABELS}; 
    MLP *mlp = createMLP(num_layers, layers_size);
    if (!mlp)printf("porcoddio\n");
    //printMLP(mlp);
    // Define learning parameters
    double learning_rate = 0.01;
    int num_epochs = 500;
    int batch_size = 128; // Adjust based on your dataset size and memory constraints

    // Train MLP
    trainMLP(splitted_dataset.train, mlp, num_epochs, batch_size, learning_rate);
    // double error = evaluateMLP(mlp,test_data,test_targets,test_size, sigmoid);
    // printf("error is %f\n",error);

    // // Clean up
    // for (int i = 0; i < n_samples; i++) {
    //     free(dataset[i]);
    //     free(targets[i]);
    // }
    // free(dataset);
    // free(targets);
    freeDataset(&splitted_dataset);
    // return 0;
}

