#ifndef THREADS_ACCOMULATORSWB.H
#define THREADS_ACCOMULATORSWB.H

#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib> // Include the <cstdlib> header to define the malloc function
#include<math.h>
#include "../include/threads_train.h"
#include "../include/mlp.h"

typedef struct {
    Thread_args_train **thread_args_train;
    MLP* mlp;
    long thread_id;
    double** weights_global; //pointer to list of weights of main thread (pointer to an array of pointers(layer) to an array of weights)
    int start_layer_weight; 
    int start_weight;
    int counter_weight_max;
    double** biases_global;  //pointer to list of weights of main thread (pointer to an array of pointers(layer) to an array of biases)
    int start_layer_bias;
    int start_bias;
    int counter_bias_max;
    int num_working_threads; //how many working threads there are (if there are more threads than wights in the NN)
} Thread_args_accomulatorWB;

int createThreadArgs_accomulatorWB(int NUM_ACC_THREADS, Thread_args_accomulatorWB* thread_args_accomulatorWB, MLP* mlp, Thread_args_train **thread_args_train, double** grad_weights_accumulators, double** grad_biases_accumulator);
void printThreadArgs_accomulatorWB(Thread_args_accomulatorWB* thread_args_accomulatorWB, int NUM_ACC_THREADS);


#endif // THREADS_ACCOMULATORSWB.H