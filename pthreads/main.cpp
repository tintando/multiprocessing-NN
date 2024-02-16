#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include "include/activation_functions.h"
#include "include/dataset.h"
#include "include/mlp.h"
#include "include/threads.h"

#define N_FEATURES 8
#define N_LABELS 1
#define NUM_THREADS 10

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

    // compute neuron activation for the hidden layers and output layer
    for (int i = 1; i < args->mlp->num_layers; i++) {
         // for each hidden layer
        matrixMultiplyAndAddBias(args->my_neuron_activations[i], args->my_neuron_activations[i-1], args->mlp->weights[i],
                                args->mlp->biases[i], args->mlp->layers_sizes[i-1], args->mlp->layers_sizes[i]);
        
        applyActivationFunction(args->my_neuron_activations[i], args->mlp->layers_sizes[i], args->act);
    }
}

void backpropagation_thread(Thread_args* args, int sample_i){
    //the loss for this sample
    double sample_loss=0.0;

    //the last layer computes error in a different way than other layers
    //iterate trough nodes of last layer
    int last_layer_index = args->mlp->num_layers-1;
    for (int j = 0; j < args->mlp->layers_sizes[last_layer_index]; j++) {

        // error = result - expected
        double output_error = args->dataset->targets[sample_i][j] - args->my_neuron_activations[last_layer_index][j];

        //delta = error * value of dact at (value of output node j)
        args->my_delta[last_layer_index][j] = output_error * args->dact(args->my_neuron_activations[last_layer_index][j]);

        //accomulating
        args->my_grad_biases_accumulator[last_layer_index][j] += args->my_delta[last_layer_index][j]; 
        
        //the square of the errors
        sample_loss+=output_error*output_error;
    }
    

    args->my_batch_loss += sample_loss;

    // compute error for other layers
    //iterate trough hidden layers backward
    for (int i = args->mlp->num_layers-2; i > 0; i--){
        //printf("\nProcessing layer %d\n", i);
        //iterate trough neurons of the layer
        for (int j = 0; j < args->mlp->layers_sizes[i]; j++) {

            //error for the layer
            double error = 0.0;
            
            // iterate trough neurons of next layer and compute total error of neuron j of layer i
            for (int k = 0; k<args->mlp->layers_sizes[i+1]; k++) {
                    error += args->mlp->weights[i+1][k * args->mlp->layers_sizes[i] + j] * args->my_delta[i+1][k];
                                    }
             //printf("Computed error for neuron %d in layer %d: %f\n", j, i, error);
             
            //compute delta for the neuron in current layer
            args->my_delta[i][j] = error * args->dact(args->my_neuron_activations[i][j]);
            // printf("Delta for neuron %d in layer %d: %f\n", j, i, args->my_delta[i][j]);
            // printf("Note: my_neuron_activations of neuron %d of layer %d is %lf\n", j, i, args->my_neuron_activations[i][j]);

            // accumulate deltas for all the samples relative to the thread 
            args->my_grad_biases_accumulator[i][j] += args->my_delta[i][j]; 

            // iterate trough neurons of previous layer
            // gradient = delta[current_layer][neuron] * previous layer neuron value
            for (int k = 0; k < args->mlp->layers_sizes[i-1]; k++) {// for each neuron in previous layer
                
                //compute gradient
                double grad = args->my_delta[i][j] * args->my_neuron_activations[i-1][k];
                // printf("Gradient for connection %d->%d in layer %d: %f", k, j, i, grad);
                // printf(" computed as %lf * %lf\n ", args->my_delta[i][j], args->my_neuron_activations[i-1][k]);
                
                // accumulate deltas for all the samples relative to the thread 
                args->my_grad_weights_accumulators[i][j * args->mlp->layers_sizes[i-1] + k] += grad;
            }
            //is this going here????
            //args->my_grad_biases_accumulator[i][j] += args->my_delta[i][j];// accumulate deltas 
        }

    }
    //printf("[%d] Backpropagation finished for sample %d\n",args->thread_id, sample_i);
}




void *thread_action(void *voidArgs){
    
    Thread_args *args = (Thread_args *)voidArgs;//casting the correct type to args
    //if (args->thread_id==3) printf("Hello world from thread %ld\n", args->thread_id);
    double batch_loss = 0;
    //if it is last thread, it has less samples, or 0 (in case it is divisible)
    int my_number_of_samples = (args->thread_id != NUM_THREADS-1) ? 
                args->batch_size/NUM_THREADS : args->batch_size/NUM_THREADS + args->batch_size % NUM_THREADS-1;

    //int my_number_of_samples = args->batch_size/NUM_THREADS; //with one thread
    int my_start_index = args->batch_start_index + args->thread_id * my_number_of_samples;
    int my_end_index = my_start_index + my_number_of_samples;
    //if (args->thread_id==2) printf("for (int sample_i = %d; sample_i<%d; sample_i++) {\n", my_start_index, my_end_index);
    
    //iterate trough my samples
    for (int sample_i = my_start_index; sample_i<my_end_index; sample_i++) {
        //set sample_i features as neuron activation of first layer
        for (int j = 0; j < args->mlp->layers_sizes[0]; j++) {
            args->my_neuron_activations[0][j] = args->dataset->samples[sample_i][j];
        }

//uncomment this if you want to see Input layer and all other layers at beginning of feed forward
// //set sample_i features as neuron activation of first layer
// for (int j = 0; j < args->mlp->layers_sizes[0]; j++) {
//     args->my_neuron_activations[0][j] = args->dataset->samples[sample_i][j];
// }
// for (int i = 1; i < args->mlp->num_layers; i++) {
//     for (int j = 0; j < args->mlp->layers_sizes[i]; j++) {
//         args->my_neuron_activations[i][j] = 0.0;// initialize
//     }
// }
        
        double sample_loss = 0;
        //printf("\n[%d]sample %d after feedforward\n",args->thread_id, sample_i);
        feedforward_thread(args);
        //printThreadArgs(args);
        // printf("\n\nstarting backpropagation\n");
        // printThreadArgs(args);
        backpropagation_thread(args, sample_i);
        //printf("\n\n%dsample %d/%d  before feedforward\n",args->thread_id, sample_i, my_end_index-1);
        

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
    
    return NULL;
}

void trainMLP(Data train_dataset, MLP* mlp, int num_epochs, int batch_size, int learning_rate){
    
    //initialize thread data structures
    pthread_t threads[NUM_THREADS]; //thread identifier
    Thread_args* thread_args[NUM_THREADS]; // array of thread data, one specific for thread

    // thread independent accomulator of gradient weights for a batch, it is gonna be resetted every batch
    // an array of pointers(layer) that point an to array of pointers(neurons of current layer) that point to an array of double (neurons of prevoius layer)
    //[layer][current_layer_neuron][previous_layer_neuron] = gradient between the neuron of current layer and a neuron of previous layer
    double **grad_weights_accumulators = (double **)malloc(mlp->num_layers * sizeof(double **));

    // thread independent accomulator of gradient weights for a batch, , it is gonna be resetted every batch
    // an array of pointers (layers) that points to an array of doubles (the bias gradients)
    //[layer][neuron]
    double **grad_biases_accumulator = (double **)malloc((mlp->num_layers) * sizeof(double *));



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
for (int epoch = 0; epoch < num_epochs; epoch++) {
        //printf("epoch %d: \n", epoch);
        double epoch_loss = 0.0; //accomulator of loss over a single epoch

        // iterate through the dataset in batches
        //train_dataset.size = 2; //tmp (specify the number of sample to try)

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


        //initialize accumulators to 0 
        for (int i = 1; i<mlp->num_layers; i++){
                grad_weights_accumulators[i] = (double *)calloc(mlp->layers_sizes[i] * mlp->layers_sizes[i-1], sizeof(double));
                grad_biases_accumulator[i] = (double *)calloc(mlp->layers_sizes[i], sizeof(double));    
            }


        // initializing data structures of threads that are dependent on the batch
        for (long thread_id = 0; thread_id < NUM_THREADS; ++thread_id) {
            
            thread_args[thread_id]->batch_size = current_batch_size;
            thread_args[thread_id]->batch_start_index = batch_start_index;

            //starting the threads
            //printf("creating thread %d\n", thread_id);
            pthread_create(&threads[thread_id], NULL,  thread_action, (void *)thread_args[thread_id]);
        }

        for(long thread_id = 0; thread_id < NUM_THREADS; thread_id++){
            pthread_join(threads[thread_id], NULL);
        }
        //printf("summing accomulators");
        //sum the accumulators of all the threads (this should be parralelized)
        for (int thread_id = 0; thread_id < NUM_THREADS; thread_id++){
            // loop trough each layer
            for (int i = 1; i < mlp->num_layers-1; i++) {
                // loop trough each neuron
                for (int j = 0; j < mlp->layers_sizes[i]; j++) {
                    grad_biases_accumulator[i][j] += thread_args[thread_id]->my_grad_biases_accumulator[i][j];
                    thread_args[thread_id]->my_grad_biases_accumulator[i][j] = 0;//resetting them
                    //summing weights accomulators
                    for (int k = 0; k < mlp->layers_sizes[i-1]; k++) {// loop trough each neuron
                        grad_weights_accumulators[i][j * mlp->layers_sizes[i-1] + k] += thread_args[thread_id]->my_grad_weights_accumulators[i][j * mlp->layers_sizes[i-1] + k];
                        thread_args[thread_id]->my_grad_weights_accumulators[i][j * mlp->layers_sizes[i-1] + k] = 0;
                    }
                }
            }
        }
        // printf("accomulators computed");

        // printf("Gradient Weights Accumulators:\n");
        // for (int i = 1; i < mlp->num_layers; i++) {
        //     printf("Layer %d: ", i);
        //     for (int j = 0; j < mlp->layers_sizes[i] * mlp->layers_sizes[i-1]; j++) {
        //         printf("%lf ", grad_weights_accumulators[i][j]);
        //     }
        //     printf("\n");
        // }
        // // Print gradient accumulators for biases
        // printf("Gradient Biases Accumulators:\n");
        // for (int i = 1; i < mlp->num_layers; i++) {
        //     printf("Layer %d: ", i);
        //     for (int j = 0; j < mlp->layers_sizes[i]; j++) {
        //         printf("%lf ", grad_biases_accumulator[i][j]);
        //     }
        //     printf("\n");

        // }

        // Apply mean gradients to update weights and biases
        for (int i = 1; i < mlp->num_layers; i++) {
            
            //for each neuron in current layer
            for (int j = 0; j < mlp->layers_sizes[i]; j++) { 

                // Calculate mean gradient for biases and update
                double mean_grad_bias = grad_biases_accumulator[i][j] / current_batch_size;
                mlp->biases[i][j] += learning_rate * mean_grad_bias;

                for (int k = 0; k < mlp->layers_sizes[i-1]; k++) {
                    // Calculate mean gradient
                    double mean_grad = grad_weights_accumulators[i][j * mlp->layers_sizes[i-1] + k] / current_batch_size;
                    // Update weights
                    mlp->weights[i][j * mlp->layers_sizes[i-1] + k] += learning_rate * mean_grad;
                }
            }
        }
    }
    //batch computed

}
}

int main(int argc, char *argv[]){

    const char* filename = "/home/lexyo/Documenti/Dev/Multicore/multiprocessing-NN/pthreads/datasets/newyork.csv";
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

