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
#define NUM_THREADS 16


typedef struct Thread_args{
    int thread_id;
    MLP* mlp;
    double **my_neuron_activations; //the neuron activations for the samples of the thread (array of pointers (layers) to array of doubles)
    double **my_delta;  //array of pointers to array of doubles
    double **my_grad_weights_accumulators;
    double **my_grad_biases_accumulator;
    int batch_start_index;
    int batch_size;
    Data* dataset; // pointer to the dataset 
    ActivationFunction act;
    ActivationFunctionDerivative dact;
    double learning_rate;
    int num_threads;
    double my_loss;
}Thread_args;


void matrixMultiplyAndAddBias(double *output, double *input, 
                              double *weights, double *biases, 
                              int inputSize, int outputSize) {
    for (int neuron_i = 0; neuron_i < outputSize; neuron_i++) {// neuron in current layer
        output[neuron_i] = 0.0;// initialize
        for (int neuron_j = 0; neuron_j < inputSize; neuron_j++) {// neuron of previous layer
            //multiply each input neuron by corresponding weight
            output[neuron_i] += input[neuron_j] * weights[neuron_i * inputSize + neuron_j]; 
        }
        output[neuron_i] += biases[neuron_i];//add bias
    }
}

void feedforward_thread(Thread_args* args){
    // gives input
   MLP *mlp = args->mlp;
    
    for (int i = 0; i < mlp->input_size; i++) {
        // Initialize the activation of the input layer neurons with the input values.
        args->my_neuron_activations[0][i] = args->dataset->samples[0][i];
    }

    // compute neuron activation for the hidden layers
    for (int current_layer = 0; current_layer < mlp->num_hidden_layers; current_layer++) {
         // for each hidden layer
        printf("neuron activation for layer /d\n", current_layer);
        matrixMultiplyAndAddBias(args->my_neuron_activations[next_layer],
                                 args->my_neuron_activations[current_layer],
                                 mlp->weights[current_layer], mlp->biases[current_layer],
                                 mlp->hidden_layers_size[current_layer],
                                 mlp->hidden_layers_size[next_layer]
        );
        
        applyActivationFunction(args->my_neuron_activations[next_layer], mlp->hidden_layers_size[next_layer], args->act);
    }
    // compute neuron activation for the output layer
    matrixMultiplyAndAddBias(args->my_neuron_activations[mlp->num_hidden_layers],
                             args->my_neuron_activations[mlp->num_hidden_layers - 1],
                             mlp->weights[mlp->num_hidden_layers - 1],
                             mlp->biases[mlp->num_hidden_layers - 1],
                             mlp->hidden_layers_size[mlp->num_hidden_layers - 1], 
                             mlp->output_size);
    applyActivationFunction(args->my_neuron_activations[mlp->num_hidden_layers], mlp->output_size, args->act);
}




void *thread_action(void *voidArgs){

    Thread_args *args = (Thread_args *)voidArgs;
    printf("Hello world from thread %d", args->thread_id);
    double batch_loss = 0;

    int number_of_samples = (args->thread_id != NUM_THREADS-1) ? args->batch_size/NUM_THREADS : args->batch_size % NUM_THREADS;
    int my_start_index = args->thread_id * number_of_samples;
    int my_end_index = my_start_index + number_of_samples;

    for (int sample= 0; sample<my_end_index-my_start_index+1; sample++) {// for each each sample in the batch

        double sample_loss = 0;
        // feedforward_thread(args);

        for (int i = 0; i < args->mlp->output_size; i++) {// for each output node
                // error = result - expected
                double output_error = args->dataset->samples[sample][i] - args->my_neuron_activations[args->mlp->num_hidden_layers][i];
                // delta = error * derivativeofactivationfunction(value_of_output_node_i)
                args->my_delta[args->mlp->num_hidden_layers][i] = output_error * args->dact(args->mlp->neuron_activations[args->mlp->num_hidden_layers][i]);
                //This step quantifies how each output neuron's activation needs to change to reduce the overall error.
                sample_loss+=output_error*output_error;
        }
        batch_loss+=sample_loss;
    }
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
Thread_args* createThreadArgs(MLP *mlp, int thread_id){
    //Thread_args structure
    Thread_args* args = (Thread_args*) malloc(sizeof(Thread_args));
    if (!args) return NULL;

    args->thread_id = thread_id;
    args->mlp = mlp;
    args->my_neuron_activations = (double**)malloc((mlp->num_hidden_layers + 1) * sizeof(double *));
    args->my_delta = (double **)malloc((mlp->num_hidden_layers + output_layer) * sizeof(double *));
    args->my_grad_weights_accumulators = (double**)malloc((mlp->num_hidden_layers + output_layer) * sizeof(double **));
    args->my_grad_biases_accumulator = (double **)malloc((mlp->num_hidden_layers + output_layer) * sizeof(double *));
    args->my_loss = 0;
    args->act = relu;
    args->dact = drelu;


    for(int i = 0; i <= mlp->num_hidden_layers; i++){
        int layer_size = (i == mlp->num_hidden_layers) ? mlp->output_size : mlp->hidden_layers_size[i];
        args->my_neuron_activations[i] = (double *)calloc(layer_size, sizeof(double));
    }

    for(int current_layer = 0; current_layer <= mlp->num_hidden_layers; current_layer++){
        int size_in = (current_layer == 0) ? mlp->input_size : mlp->hidden_layers_size[previous_layer];
        int size_out = (current_layer == mlp->num_hidden_layers) ? mlp->output_size : mlp->hidden_layers_size[current_layer];
        args->my_grad_weights_accumulators[current_layer] = (double *)malloc(size_out * size_in * sizeof(double *));
        args->my_grad_biases_accumulator[current_layer] = (double *)calloc(size_out, sizeof(double));
    }
    //hidden={2,4}
    for(int layer = 0; layer <= mlp->num_hidden_layers; layer++){
        int layer_size = (layer == mlp->num_hidden_layers) ? mlp->output_size : mlp->hidden_layers_size[layer];
        args->my_delta[layer] = (double *)malloc(layer_size * sizeof(double));
    }


    //if there were errors, free everything
    if(!args->my_neuron_activations || !args->my_grad_weights_accumulators || !args->my_grad_biases_accumulator || !args->my_delta){

        for (int i = 0; i <= mlp->num_hidden_layers; i++) {
            free(args->my_neuron_activations[i]);
            free(args->my_delta[i]);
        }
        free(args->my_neuron_activations);
        for (int i = 0; i <= mlp->num_hidden_layers; i++) {
            free(args->my_grad_weights_accumulators[i]);
            free(args->my_grad_weights_accumulators[i]);
            free(args->my_grad_biases_accumulator[i]);
        }
        free(args->my_grad_weights_accumulators);
        free(args->my_grad_biases_accumulator);
        free(args);
        return NULL;
    }

    return args;
}

void trainMLP(Data train_dataset, MLP* mlp, int num_epochs, int batch_size, int learning_rate){
    
    //initialize thread data structures
    pthread_t threads[NUM_THREADS]; //thread identifier
    Thread_args* thread_args[NUM_THREADS]; // array of thread data, one specific for thread
    
    long thread;

    //Initializes the variables that are persistent in the epochs or while iterating the samples
    for(thread=0; thread < NUM_THREADS; thread++){
        thread_args[thread] = createThreadArgs(mlp,thread); 
        thread_args[thread]->dataset = &train_dataset;
    }

    //for each epoch
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("epoch %d: \n", epoch);
        double epoch_loss = 0.0; //accomulator of loss over a single epoch
        printData(train_dataset);
        // for each batch
        printf("%d", train_dataset.size);
         for (int batch_start_index = 0; batch_start_index < train_dataset.size; batch_start_index += batch_size) { // iterate through the dataset in batches.
            printData(train_dataset);
            printf("%d",batch_start_index);
            printf("batch starting from %d\n", batch_start_index);
            //the size of the ith batch.
            printf("dsadsda%d",batch_start_index);
            printf("%d", train_dataset.size);
            int current_batch_size = (batch_start_index + batch_size > train_dataset.size) ? (train_dataset.size - batch_start_index) : batch_size;
            printf("hello");
            // double **batch_inputs = (double **)malloc(current_batch_size * sizeof(double *));// the inputs for this batch
            // double **batch_targets = (double **)malloc(current_batch_size * sizeof(double *));// the labels for this batch

            // for (int j = 0; j < current_batch_size; j++) {
            //         batch_inputs[j] = dataset[i + j];
            //         batch_targets[j] = targets[i + j];
            //     }


            //initializing data structures of threads that are dependent on the batch
            for (int thread_id = 0; thread_id < NUM_THREADS; ++thread_id) {
                
                thread_args[thread_id]->batch_size = current_batch_size;
                thread_args[thread_id]->batch_start_index = batch_start_index;

                printf("starting threads");
                //starting the threads
                pthread_create(&threads[thread_id], NULL,  thread_action, (void *)&thread_args[thread_id]);
            }

            for(int thread_id = 0; thread_id < NUM_THREADS; thread_id++){
                pthread_join(threads[thread_id], NULL);
            }

        }
    }
    
}

int main(int argc, char *argv[]){

    const char* filename = "/home/lexyo/Documenti/Dev/Multicore/multiprocessing-NN/pthreads/datasets/california.csv";
    double **dataset = NULL, **targets = NULL;
    int n_samples = 0;

    // Load and prepare the dataset
    loadAndPrepareDataset(filename, &dataset, &targets, &n_samples);
    
    Dataset splitted_dataset = splitDataset(n_samples, &dataset, &targets);

    //can be freed, we don't need them anymore
    free(dataset);
    free(targets);

    printData(splitted_dataset.train);
    // Initialize your MLP
    int input_size = N_FEATURES; // Define according to your dataset
    int output_size = N_LABELS; // Typically 1 for regression tasks
    int num_hidden_layers = 2; // Example: 2 hidden layers
    int hidden_layers_size[] = {5, 2}; // Example sizes for the hidden layers
    MLP *mlp = createMLP(input_size, output_size, num_hidden_layers, hidden_layers_size);

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

    // return 0;
}