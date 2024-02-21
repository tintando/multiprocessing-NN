#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib> // Include the <cstdlib> header to define the malloc function
#include<math.h>
#include "../include/threads_train.h"
#include "../include/mlp.h"

typedef struct {
    Thread_args_train **thread_args_train;
    long thread_id;
    double*** weights_global; //pointer to list of weights of main thread (pointer to an array of pointers(layer) to an array of weights)
    int start_layer_weight; 
    int start_weight;
    int counter_weight_max;
    double*** biases_global;  //pointer to list of weights of main thread (pointer to an array of pointers(layer) to an array of biases)
    int start_layer_bias;
    int start_bias;
    int counter_bias_max;
    int num_layers; 
    int* layers_size; //pointer to array of pointers(layer) to layers sizes
    int num_working_threads; //how many working threads there are (if there are more threads than wights in the NN)
} Thread_args_accomulatorWB;

int createThreadArgsSum(int NUM_ACC_THREADS, Thread_args_accomulatorWB* thread_args_accomulatorWB, MLP* mlp, Thread_args_train **thread_args_train, double*** grad_weights_accumulators, double***grad_biases_accumulator){

    //compute amount of weights in the network
    int total_weights = 0;
    for(int i = 1; i < mlp->num_layers; i++){
        total_weights += mlp->layers_sizes[i] * mlp->layers_sizes[i-1];
        //printf("layer[%d] has %d weights connecting to previous layer\n", i, mlp->layers_sizes[i] * mlp->layers_sizes[i-1]);
    } //total weights = # of weights of all layers

    //compute amount of weights per thread and remainder
    int weights_per_thread = total_weights/NUM_ACC_THREADS;
    //printf("weights_per_thread = %d\n", weights_per_thread);
    int remainder_weights = total_weights%NUM_ACC_THREADS;
    //printf("remainder = %d\n", remainder_weights);

    int start_flag = 1; //tells if the current thread should store the current index as a start index
    int thread_id = 0; //id of the accomulator thread
    int count = 0; // keeps track of how many weight the acc thread has taken till now
    //loop trough the weight structure of the network and assign the start index for weight to each thread
    for (int i = 1; i<mlp->num_layers; i++){
        for (int w=0; w < mlp->layers_sizes[i] * mlp->layers_sizes[i-1]; w++){//iterate trough the weights
            if (start_flag) {
                printf("weights thread %d starts from [%d][%d]\n", thread_id, i, w); 
                thread_args_accomulatorWB[thread_id].start_layer_weight = i;
                thread_args_accomulatorWB[thread_id].start_weight = w;
                start_flag = 0;
            }
            count++;
            if (count >= ((remainder_weights>0) ? weights_per_thread + 1 : weights_per_thread)){
                printf("weights thread %d finishs at [%d][%d], total of %d weights\n", thread_id, i, w, count);
                thread_args_accomulatorWB[thread_id].counter_weight_max = count;
                start_flag = 1;
                remainder_weights --;
                thread_id ++;
                count = 0;
            }
        }
    }

    int num_working_weights_threads = thread_id;
    for (int i = 0; i < NUM_ACC_THREADS; i++){
        if (i >= num_working_weights_threads) thread_args_accomulatorWB[i].start_layer_weight = -1;
        thread_args_accomulatorWB[i].num_working_threads = num_working_weights_threads;
    }


    //compute amount of bias in the network
    int total_biases = 0;
    for(int i = 1; i < mlp->num_layers; i++){
        total_biases += mlp->layers_sizes[i];
        //printf("layer[%d] has %d biases\n", i, mlp->layers_sizes[i]);
    } //total biases = # of biases of all layers

    //compute amount of biases per thread and remainder
    int biases_per_thread = total_biases/NUM_ACC_THREADS;
    //printf("biases_per_thread = %d\n", biases_per_thread);
    int remainder_biases = total_biases%NUM_ACC_THREADS;
    //printf("remainder_biases = %d\n", remainder_biases);

    thread_id = 0;
    count = 0;
    start_flag = 1;
    for (int i = 1; i<mlp->num_layers; i++){
        for (int j=0; j < mlp->layers_sizes[i]; j++){//iterate trough the biases
            if (start_flag) {
                //printf("biases thread %d starts from [%d][%d]\n", thread_id, i, j); 
                thread_args_accomulatorWB[thread_id].start_layer_bias = i;
                thread_args_accomulatorWB[thread_id].start_bias = j;
                start_flag = 0;
            }
            count++;
            if (count >= ((remainder_biases>0) ? biases_per_thread + 1 : biases_per_thread)){
                //printf("biases thread %d finishs at [%d][%d], total of %d bias\n", thread_id, i, j, count);
                thread_args_accomulatorWB[thread_id].counter_bias_max = count;
                start_flag = 1;
                remainder_biases --;
                thread_id ++;
                count = 0;
            }
        }
    }
    //if there are more threads than biases, ses the start biase,layer index to -1, so that that thread doesn0t do anything
    if (thread_id<NUM_ACC_THREADS){
        for (int i = thread_id; i<NUM_ACC_THREADS; i++){
            thread_args_accomulatorWB[i].start_layer_bias = -1;
        }
    }  
//------------each thread now knows the range of weights and biases it has to sum----------------

//------------syncronize accomulator threads with feedforward and backpropagation threads----------------
    for (int i = 0; i < NUM_ACC_THREADS; i++){
        thread_args_accomulatorWB[i].layers_size = mlp->layers_sizes;
        thread_args_accomulatorWB[i].num_layers = mlp->num_layers;
        thread_args_accomulatorWB[i].thread_id = i; //the logical id of the thread
        thread_args_accomulatorWB[i].thread_args_train = thread_args_train; //the pointer of the list of pointers to the backpropagation and feedforward threads
        thread_args_accomulatorWB[i].weights_global = grad_weights_accumulators; //the pointer to the list of weights of the main thread
        thread_args_accomulatorWB[i].biases_global = grad_biases_accumulator;   //the pointer to the list of biases of the main thread
    }

    printf("check of weights and biases\n");
    double weight = thread_args_accomulatorWB[0].thread_args_train[0]->my_grad_weights_accumulators[1][0];  
    printf("----------------------%lf\n", weight);

    return num_working_weights_threads;
}

