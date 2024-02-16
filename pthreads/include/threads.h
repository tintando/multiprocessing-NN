#ifndef THREADS_H
#define THREADS_H

#include "mlp.h"
#include "activation_functions.h"
#include "dataset.h"

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

Thread_args* createThreadArgs(MLP *mlp, long thread_id);

#endif // THREADS_H