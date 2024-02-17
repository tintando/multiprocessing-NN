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
#define NUM_THREADS 2

//---------------------train-------------------

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
        //printf("[%d]sample i: [%d], target: %f, output: %f, error: %f\n",args->thread_id, sample_i, args->dataset->targets[sample_i][j], args->my_neuron_activations[last_layer_index][j], output_error);
        for (int k = 0; k < args->mlp->layers_sizes[last_layer_index-1]; k++) {
            //gradient = delta * previous layer neuron value
            double grad = args->my_delta[last_layer_index][j] * args->my_neuron_activations[last_layer_index-1][k];
            args->my_grad_weights_accumulators[last_layer_index][j * args->mlp->layers_sizes[last_layer_index-1] + k] += grad;
        }
        
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
    // printf("\nGradient Weights Accumulators:\n");
    // for (int i = 1; i < args->mlp->num_layers; i++) {
    //     for (int j = 0; j < args->mlp->layers_sizes[i]; j++) {
    //         for(int k = 0; k < args->mlp->layers_sizes[i-1]; k++){
    //             printf("neuron %d of layer %d to neuron %d of layer %d: %lf ",k, i-1, j, i, args->my_grad_weights_accumulators[i][j * args->mlp->layers_sizes[i-1] + k]);
    //             printf("\n");
    //         }
    //     }
    //     printf("\n");
    // }
    // // Print gradient accumulators for biases
    // printf("Gradient Biases Accumulators:\n");
    // for (int i = 1; i < args->mlp->num_layers; i++) {
    //     printf("Layer %d: ", i);
    //     for (int j = 0; j < args->mlp->layers_sizes[i]; j++) {
    //         printf("%lf ", args->my_grad_biases_accumulator[i][j]);
    //     }
    //     printf("\n");
    // }
}




void *thread_action_train(void *voidArgs){
    
    Thread_args *args = (Thread_args *)voidArgs;//casting the correct type to args
    //if (args->thread_id==3) printf("Hello world from thread %ld\n", args->thread_id);
    double batch_loss = 0;
    //if it is last thread, it might have less samples, (the same number of other threads if the number of threads is divisor of batch size)
    int my_number_of_samples = (args->thread_id == NUM_THREADS-1) ? args->batch_size/NUM_THREADS - args->batch_size%NUM_THREADS : args->batch_size/NUM_THREADS;

    //int my_number_of_samples = args->batch_size/NUM_THREADS; //with one thread
    //printf("[%d] i have %d samples, in particular from %d to %d, batch size = %d --- batch start index = %d \n", args->thread_id, my_number_of_samples, args->batch_start_index + args->thread_id * my_number_of_samples, args->batch_start_index + args->thread_id * my_number_of_samples + my_number_of_samples, args->batch_size, args->batch_start_index);
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
        //printf("\n\n[%d] sample %d/%d  starting feedforward and backprop\n",args->thread_id, sample_i, my_end_index-1);
        double sample_loss = 0;
        //printf("\n[%d]sample %d after feedforward\n",args->thread_id, sample_i);
        feedforward_thread(args);
        //printThreadArgs(args);
        // printf("\n\nstarting backpropagation\n");
        // printThreadArgs(args);
        backpropagation_thread(args, sample_i);
        //printf("\n\n%dsample %d/%d  before feedforward\n",args->thread_id, sample_i, my_end_index-1);
        //printf("[%d] sample %d/%d  finished feedforward and backprop\n",args->thread_id, sample_i, my_end_index-1);
    }
    //the average of the errors of this thread over its split of the batches
    args->my_batch_loss = args->my_batch_loss/my_number_of_samples;
    
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

    int current_batch_size;

    //for each epoch
    for (int epoch = 0; epoch < num_epochs; epoch++) {
            printf("epoch %d: \n", epoch);
            double epoch_loss = 0.0; //accomulator of loss over a single epoch

            // iterate through the dataset in batches
            //train_dataset.size = 2; //tmp (specify the number of sample to try)

        for (int batch_start_index = 0; batch_start_index < train_dataset.size; batch_start_index += batch_size) { 
            //printData(train_dataset);
            //the size of the ith batch.
            double batch_loss = 0;
            if (batch_start_index + batch_size > train_dataset.size){
                current_batch_size = train_dataset.size - batch_start_index;
                //printf("batch start: %d, batch end(in theory) = %d, SPECIAL size of this batch: %d --- \n", batch_start_index, batch_start_index + batch_size, current_batch_size);

            }
            else{
                current_batch_size = batch_size;
                //printf("batch start: %d, batch end = %d, normal size of this batch: %d\n", batch_start_index, batch_start_index + batch_size, current_batch_size);

            }
            //int current_batch_size = (batch_start_index + batch_size > train_dataset.size) ? (batch_start_index + batch_size - train_dataset.size) : batch_size;
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
                pthread_create(&threads[thread_id], NULL,  thread_action_train, (void *)thread_args[thread_id]);
            }

            for(long thread_id = 0; thread_id < NUM_THREADS; thread_id++){
                pthread_join(threads[thread_id], NULL);
            }

            //printf("summing accomulators");
            //sum the accumulators of all the threads (this should be parralelized)
            for (int thread_id = 0; thread_id < NUM_THREADS; thread_id++){
                // loop trough each layer
                //printf("\nthread[%d] loss: %f", thread_id, thread_args[thread_id]->my_batch_loss);
                batch_loss += thread_args[thread_id]->my_batch_loss;
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
            // printf("accomulators computed")

            // printf("\nBATCH Gradient Weights Accumulators:\n");
            // for (int i = 1; i < mlp->num_layers; i++) {
            //     for (int j = 0; j < mlp->layers_sizes[i]; j++) {
            //         for(int k = 0; k < mlp->layers_sizes[i-1]; k++){
            //             printf("neuron %d of layer %d to neuron %d of layer %d: %lf ",k, i-1, j, i, grad_weights_accumulators[i][j * mlp->layers_sizes[i-1] + k]);
            //             printf("\n");
            //         }
            //     }
            //     printf("\n");
            // }
            // // Print gradient accumulators for biases
            // printf("BATCH Gradient Biases Accumulators:\n");
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
                    double mean_grad_bias = grad_biases_accumulator[i][j] / (train_dataset.size/NUM_THREADS);
                    // printf("bias of neuron %d of layer %d: %lf ", j, i, mlp->biases[i][j]);
                    // printf("mean_grad_bias: %lf", mean_grad_bias);
                    // printf(" new bias: %lf\n", mlp->biases[i][j] + learning_rate * mean_grad_bias);
                    mlp->biases[i][j] += learning_rate * mean_grad_bias;

                    for (int k = 0; k < mlp->layers_sizes[i-1]; k++) {
                        // Calculate mean gradient
                        double mean_grad = grad_weights_accumulators[i][j * mlp->layers_sizes[i-1] + k] / (train_dataset.size/NUM_THREADS);
                        // printf("weight of neuron %d of layer %d to neuron %d of layer %d: %lf ", k, i-1, j, i, mlp->weights[i][j * mlp->layers_sizes[i-1] + k]);
                        // printf("mean_grad: %lf", mean_grad);
                        // printf(" new weight: %lf\n", mlp->weights[i][j * mlp->layers_sizes[i-1] + k] + learning_rate * mean_grad);
                        // Update weights
                        mlp->weights[i][j * mlp->layers_sizes[i-1] + k] += learning_rate * mean_grad;
                    }
                }
            }
            //batch_loss
            batch_loss = batch_loss/NUM_THREADS;
            //printf("\nbatch loss: %f\n", batch_loss);
            epoch_loss += batch_loss;
        }
        // all batches of this epoch  computed
        epoch_loss = epoch_loss/(train_dataset.size/batch_size);
        printf("epoch_loss: %f\n", epoch_loss);
    }
    //all epochs done
}
//-----------------end train-------------------

//----------------evaluation------------------
void *thread_action_evaluation(void *voidArgs){
    //the thread thinks the whole train dataset is a batch, so it will split it evenly
    Thread_args *args = (Thread_args *)voidArgs;//casting the correct type to args

    //if it is last thread, it might have less samples, (the same number of other threads if the number of threads is divisor of batch size)
    int my_number_of_samples = (args->thread_id == NUM_THREADS-1) ? args->batch_size/NUM_THREADS - args->batch_size%NUM_THREADS : args->batch_size/NUM_THREADS;
    printf("[%d]my_number_of_samples = %d\n",args->thread_id, my_number_of_samples);
    //int my_number_of_samples = args->batch_size/NUM_THREADS; //with one thread
    //printf("[%d] i have %d samples, in particular from %d to %d, batch size = %d --- batch start index = %d \n", args->thread_id, my_number_of_samples, args->batch_start_index + args->thread_id * my_number_of_samples, args->batch_start_index + args->thread_id * my_number_of_samples + my_number_of_samples, args->batch_size, args->batch_start_index);
    int my_start_index = args->batch_start_index + args->thread_id * my_number_of_samples;
    int my_end_index = my_start_index + my_number_of_samples;
    printf("[%d] i have %d samples, in particular from %d to %d, batch size = %d --- batch start index = %d \n", args->thread_id, my_number_of_samples, my_start_index, my_end_index, args->batch_size, args->batch_start_index);

    for (int sample_i = my_start_index; sample_i<my_end_index; sample_i++) {
        //set sample_i features as neuron activation of first layer
        for (int j = 0; j < args->mlp->layers_sizes[0]; j++) {
            args->my_neuron_activations[0][j] = args->dataset->samples[sample_i][j];
        }

        double sample_loss = 0;
        feedforward_thread(args);

        // Assuming the last layer's activations are the predictions
        int last_layer_index = args->mlp->num_layers-1;
        for (int j = 0; j < args->mlp->layers_sizes[last_layer_index]; j++) {    
            // Calculate error for this sample
            double error = args->dataset->targets[sample_i][j] - args->my_neuron_activations[last_layer_index][j];
            printf("[%d] target:%f activation:%f error = %f\n",args->thread_id, args->dataset->targets[sample_i][j], args->my_neuron_activations[last_layer_index][j], args->thread_id, error);
            sample_loss += error * error; // For MSE, sum the squared error
        }
        args->my_batch_loss += sample_loss;
    }
    return NULL;
}

double evaluateMLP(MLP *mlp, Data test_data, ActivationFunction act) {
    double total_error = 0.0;
    printf("STARTING EVALUATION: test_data.size = %d\n", test_data.size);
    //threads will split the train dataset and each compute their samples
    pthread_t threads[NUM_THREADS]; //thread identifier
    Thread_args* thread_args[NUM_THREADS]; // array of thread data, one specific for thread

    for(long thread_id=0; thread_id < NUM_THREADS; thread_id++){
        printf("creating thread %d\n", thread_id);
        thread_args[thread_id] = createThreadArgs(mlp,thread_id); 
        thread_args[thread_id]->dataset = &test_data;

        //the threads think the whole train dataset is a batch, so they will split it evenly
        thread_args[thread_id]->batch_size = test_data.size;
        thread_args[thread_id]->batch_start_index = 0;
        printf("thread_args[%d]->batch_size = %d\n", thread_id, thread_args[thread_id]->batch_size);
        pthread_create(&threads[thread_id], NULL,  thread_action_evaluation, (void *)thread_args[thread_id]);
        printf("thread %d created\n", thread_id);
    }

    for(long thread_id = 0; thread_id < NUM_THREADS; thread_id++){
        pthread_join(threads[thread_id], NULL);
    }

    for(long thread_id = 0; thread_id < NUM_THREADS; thread_id++){
        total_error += thread_args[thread_id]->my_batch_loss;//this sum can be done in parallel
        printf("thread_args[%d]->my_batch_loss = %f\n", thread_id, thread_args[thread_id]->my_batch_loss);
    }

    // Return average MSE over the test set
    printf("total_error = %f\n", total_error);
    return total_error / (test_data.size * mlp->layers_sizes[mlp->num_layers-1]);
}
//-------------------------------end evaluation------------------------

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
    if (!mlp) return 1;
    //printMLP(mlp);
    // Define learning parameters
    double learning_rate = 0.01;
    int num_epochs = 10000;
    int batch_size = 128; // Adjust based on your dataset size and memory constraints
    if (batch_size < NUM_THREADS){
        printf("Impossible to start the program, batch_size[%d] < num_thread[%d]", batch_size, NUM_THREADS);
        return 1;
    }
    // Train MLP
    trainMLP(splitted_dataset.train, mlp, num_epochs, batch_size, learning_rate);
    
    //double error = evaluateMLP(mlp, splitted_dataset.test, sigmoid);
    //printf("error is %f\n",error);

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

