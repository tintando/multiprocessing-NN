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
#define NUM_THREADS 4

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
    //------------COMMPUTE LOSS------------------

    // how much the result of feedforward is far from the target
    double sample_loss=0.0;
    //the loss for neuron_j
    double loss_j;
    double dloss_j;

    //iterate trough nodes of last layer
    int last_layer_index = args->mlp->num_layers-1;
    for (int j = 0; j < args->mlp->layers_sizes[last_layer_index]; j++) {
        /*
        * Calculate the squared error loss for a single output neuron j for sample_i
        * The loss function is: L = 1/2 * (target - output)^2
        * This line accumulates the squared error over all output neurons for the given sample
        */ 
        sample_loss += pow(args->dataset->targets[sample_i][j] - args->my_neuron_activations[last_layer_index][j], 2);

        /*
        * Calculate the gradient (delta) for backpropagation for the last layer's neurons
        * 
        * The derivative of the squared loss function with respect to the neuron's activation is: dL/dy = (output - target)
        * The gradient needs to account for the derivative of the activation function as well, hence the use of the chain rule.
        * 
        * By the chain rule, the total derivative is: dL/dy * dy/da = (output - target) * dact(output)
        * where dact(output) is the derivative of the activation function with respect to its input (the neuron's activation).
        * 
        * This gradient (delta) is used to update the weights in the network during backpropagation.
        */
        args->my_delta[last_layer_index][j] = (args->dataset->targets[sample_i][j] - args->my_neuron_activations[last_layer_index][j]) * args->dact(args->my_neuron_activations[last_layer_index][j]);

        //accomulating the derivative of the activation function for the last layer neurons for all the samples relative to the thread
        args->my_grad_biases_accumulator[last_layer_index][j] += args->my_delta[last_layer_index][j]; 
        
        for (int k = 0; k < args->mlp->layers_sizes[last_layer_index-1]; k++) {
            /*
            * Calculate the gradient for updating the weights connecting the penultimate layer to the last layer.
            *
            * This step is part of the backpropagation process where we adjust the weights based on the error calculated at the output layer.
            * The gradient for each weight is calculated as the product of the delta of the neuron in the last layer and the activation of the neuron in the penultimate layer.
            * 
            * Here, 'args->my_delta[last_layer_index][j]' is the delta for neuron j in the last layer, which represents the error signal propagated back from the output.
            * 'args->my_neuron_activations[last_layer_index-1][k]' is the activation of neuron k in the penultimate layer.
            * 
            * The product of these two values gives the gradient of the loss function with respect to the weight connecting neuron k in the penultimate layer to neuron j in the last layer.
            * This gradient is then used to update the weight in the gradient descent step.
            */
            double grad = args->my_delta[last_layer_index][j] * args->my_neuron_activations[last_layer_index-1][k];

            args->my_grad_weights_accumulators[last_layer_index][j * args->mlp->layers_sizes[last_layer_index-1] + k] += grad;
        }
        //finish iteration of neurons of last layer    
    }
    // Accumulate the sample loss to the batch loss for averaging total loss computation later.
    args->my_batch_loss += sample_loss;

    /*
    * Begin backpropagation for hidden layers:
    *
    * For each hidden layer, starting from the last hidden layer and moving towards the input layer, backpropagation calculates the 
    * "delta" or error derivative for each neuron. This is achieved by taking into account the error propagated back from the layer 
    * ahead (further towards the output) and the derivative of the activation function used in the neuron. This process effectively
    * distributes the responsibility for the output error across all neurons in the network, based on their contribution to that error.
    *
    * Specifically, for a neuron j in layer i, the error (delta) is calculated as follows:
    * 1. Sum the product of the weights connecting neuron j to each neuron in the next layer (i+1) and the delta of these next-layer neurons.
    *    This sums up the error contribution from neuron j to the neurons in layer i+1.
    * 2. Multiply this sum by the derivative of the activation function evaluated at the neuron's activation. This scales the error by how
    *    much neuron j's activation contributed to the error, considering the non-linearity of the activation function.
    *
    * Once the deltas are computed for all neurons in a layer, they are used to compute the gradients of the loss function with respect
    * to the weights and biases. These gradients indicate the direction in which the weights and biases should be adjusted to reduce the
    * loss, and are accumulated over all samples in a batch (or over the entire dataset, depending on the training approach).
    */
    for (int i = args->mlp->num_layers-2; i > 0; i--) {

        for (int j = 0; j < args->mlp->layers_sizes[i]; j++) {
            // Initialize the error for the current neuron j in layer i.
            double error = 0.0;
            
            // Accumulate the error from all neurons in the next layer (i+1) that are connected to the current neuron.
            for (int k = 0; k < args->mlp->layers_sizes[i+1]; k++) {
                error += args->mlp->weights[i+1][k * args->mlp->layers_sizes[i] + j] * args->my_delta[i+1][k];
            }
            // The error variable now contains the total error propagated back to neuron j in layer i from the layer ahead.
            
            // Calculate the delta for the current neuron j in layer i by multiplying the error by the derivative of the activation function.
            args->my_delta[i][j] = error * args->dact(args->my_neuron_activations[i][j]);

            // Accumulate the gradient of biases for the neuron j in layer i, to be used for bias updates.
            args->my_grad_biases_accumulator[i][j] += args->my_delta[i][j];

            // This loop calculates the gradient of weights between each neuron in the previous layer (i-1) and the current neuron j in layer i. 
            // The gradient calculation is crucial for the backpropagation algorithm, as it determines how much the weights should be adjusted during the training process.
            // The gradient ('grad') is calculated by multiplying the delta of the current neuron j (the error derivative of the neuron's output) by the activation of the neuron k in the previous layer. 
            // This product gives the partial derivative of the loss function with respect to the weight connecting neuron k in layer (i-1) to neuron j in layer i, indicating how changes to this weight would affect the overall error.
            for (int k = 0; k < args->mlp->layers_sizes[i-1]; k++) {
                
                // Compute the gradient for the weight connecting neuron k in layer (i-1) to neuron j in layer i.
                double grad = args->my_delta[i][j] * args->my_neuron_activations[i-1][k];

                // Accumulate the gradients of weights for updates, using the calculated gradient.
                args->my_grad_weights_accumulators[i][j * args->mlp->layers_sizes[i-1] + k] += grad;
            }
            // Completed processing all connections from neurons in the previous layer to neuron j in the current layer.
        }
        // Completed processing all neurons in layer i.
    }
    // Completed backward propagation for all layers.
}

void *thread_action_train(void *voidArgs){
    
    Thread_args *args = (Thread_args *)voidArgs;//casting the correct type to args

    //resetting the batch loss at beginning of every batch
    args->my_batch_loss = 0.0;
    //resetting the accumulators at beginning of every batch
    for (int i = 1; i < args->mlp->num_layers; i++) {
        for (int j = 0; j < args->mlp->layers_sizes[i]; j++) {
            args->my_grad_biases_accumulator[i][j] = 0.0;
            for (int k = 0; k < args->mlp->layers_sizes[i-1]; k++) {
                args->my_grad_weights_accumulators[i][j * args->mlp->layers_sizes[i-1] + k] = 0.0;
            }
        }
    }

    //determine the samples of the batch for this thread
    //if it is last thread, it might have less samples, (the same number of other threads if the number of threads is divisor of batch size)
    int my_number_of_samples = (args->thread_id == NUM_THREADS-1) ? args->batch_size/NUM_THREADS - args->batch_size%NUM_THREADS : args->batch_size/NUM_THREADS;

    int my_start_index = args->batch_start_index + args->thread_id * my_number_of_samples;
    int my_end_index = my_start_index + my_number_of_samples;
    
    //apply feedforward and backpropagation to each sample of the batch
    for (int sample_i = my_start_index; sample_i<my_end_index; sample_i++) {

        //set sample_i features as neuron activation of first layer
        for (int j = 0; j < args->mlp->layers_sizes[0]; j++) {
            args->my_neuron_activations[0][j] = args->dataset->samples[sample_i][j];
        }

        ////uncomment this if you want to see Input layer and all other layers at beginning of feed forward
        // //set sample_i features as neuron activation of first layer
        // for (int j = 0; j < args->mlp->layers_sizes[0]; j++) {
        //     args->my_neuron_activations[0][j] = args->dataset->samples[sample_i][j];
        // }
        // for (int i = 1; i < args->mlp->num_layers; i++) {
        //     for (int j = 0; j < args->mlp->layers_sizes[i]; j++) {
        //         args->my_neuron_activations[i][j] = 0.0;// initialize
        //     }
        // }

        feedforward_thread(args);
        backpropagation_thread(args, sample_i);

    }
    
    return NULL;
}

void trainMLP(Data train_dataset, MLP* mlp, int num_epochs, int batch_size, double learning_rate){
    
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
    }

    int current_batch_size;
    
    //double batch_loss;
    //train_dataset.size = 2; //tmp (specify the number of sample to try)
    //for each epoch
    for (int epoch = 0; epoch < num_epochs; epoch++) {
            shuffleDataset(&(train_dataset.samples), &(train_dataset.targets), train_dataset.size);
            printf("epoch %d: \n", epoch);
            double epoch_loss = 0.0; //accomulator of loss over a single epoch

            // iterate through the dataset in batches
        
        for (int batch_start_index = 0; batch_start_index < train_dataset.size; batch_start_index += batch_size) { 
            //printData(train_dataset);
            //the size of the ith batch.
            
            //current_batch_size =  (batch_start_index + batch_size > train_dataset.size) ? (train_dataset.size - batch_start_index) :  batch_size;
            if(batch_start_index + batch_size > train_dataset.size){
                current_batch_size = train_dataset.size - batch_start_index;
            }
            else{
                current_batch_size = batch_size;
            }
            //printf("current_batch_size = %d\n", current_batch_size);

            for (int i = 1; i<mlp->num_layers; i++){
                    grad_weights_accumulators[i] = (double *)calloc(mlp->layers_sizes[i] * mlp->layers_sizes[i-1], sizeof(double));
                    grad_biases_accumulator[i] = (double *)calloc(mlp->layers_sizes[i], sizeof(double));    
                }


            // initializing data structures of threads that are dependent on the batch
            for (long thread_id = 0; thread_id < NUM_THREADS; ++thread_id) {
                
                thread_args[thread_id]->batch_size = current_batch_size;
                thread_args[thread_id]->batch_start_index = batch_start_index;
                //starting the threads
                pthread_create(&threads[thread_id], NULL,  thread_action_train, (void *)thread_args[thread_id]);
            }

            for(long thread_id = 0; thread_id < NUM_THREADS; thread_id++){
                pthread_join(threads[thread_id], NULL);
            }
            // printf("batch start index + batch size = %d\n", batch_start_index + batch_size);
            // printf("talking about batch %d/%d, batch size: %d\n", batch_start_index,train_dataset.size, current_batch_size);
            //printf("summing accomulators");
            //sum the accumulators of all the threads (this should be parralelized)
            for (int thread_id = 0; thread_id < NUM_THREADS; thread_id++){
                //summing batch loss
                epoch_loss += thread_args[thread_id]->my_batch_loss;
                for (int i = 1; i < mlp->num_layers; i++) {
                    // loop trough each neuron of the layer
                    for (int j = 0; j < mlp->layers_sizes[i]; j++) {
                        //summing biases accomulators
                        grad_biases_accumulator[i][j] += thread_args[thread_id]->my_grad_biases_accumulator[i][j];
                        // printf("[%d]my_grad_biases_accumulator[%d][%d] = %f\n",thread_id, i, j, thread_args[thread_id]->my_grad_biases_accumulator[i][j]);
                        for (int k = 0; k < mlp->layers_sizes[i-1]; k++) {
                            //summing weights accomulators
                            grad_weights_accumulators[i][j * mlp->layers_sizes[i-1] + k] += thread_args[thread_id]->my_grad_weights_accumulators[i][j * mlp->layers_sizes[i-1] + k];
                            // printf("[%d]my_grad_weights_accumulators[%d][%d] = %f\n",thread_id, i, j * mlp->layers_sizes[i-1] + k, thread_args[thread_id]->my_grad_weights_accumulators[i][j * mlp->layers_sizes[i-1] + k]);
                        }
                    }
                }
            }

            // Apply mean gradients to update weights and biases
            for (int i = 1; i < mlp->num_layers; i++) {
                // loop trough each neuron of the layer
                for (int j = 0; j < mlp->layers_sizes[i]; j++) {
                    // Calculate mean gradient for biases and update
                    // printf("grad_biases_accumulator[%d][%d] = %f\n", i, j, grad_biases_accumulator[i][j]/current_batch_size);
                    // printf("mlp->biases[%d][%d] = %f\n", i, j, mlp->biases[i][j]);
                    mlp->biases[i][j] += learning_rate * (grad_biases_accumulator[i][j] / current_batch_size);
                    // printf("mlp->biases[%d][%d] = %f\n", i, j, mlp->biases[i][j]);
                    for (int k = 0; k < mlp->layers_sizes[i-1]; k++) {
                        //calcuate mean gradient for weights and update
                        // printf("grad_weights_accumulators[%d][%d] = %f\n", i, j * mlp->layers_sizes[i-1] + k, grad_weights_accumulators[i][j * mlp->layers_sizes[i-1] + k]/batch_size);
                        // printf("mlp->weights[%d][%d] = %f\n", i, j * mlp->layers_sizes[i-1] + k, mlp->weights[i][j * mlp->layers_sizes[i-1] + k]);
                        mlp->weights[i][j * mlp->layers_sizes[i-1] + k] += learning_rate * (grad_weights_accumulators[i][j * mlp->layers_sizes[i-1] + k] / current_batch_size);
                        // printf("mlp->weights[%d][%d] = %f\n", i, j * mlp->layers_sizes[i-1] + k, mlp->weights[i][j * mlp->layers_sizes[i-1] + k]);
                        // print learning rate, grad weight accumulator, current batch size and learning_rate * (grad_weights_accumulators[i][j * mlp->layers_sizes[i-1] + k] / current_batch_size
                        //printf("learning rate = %f, grad weight accumulator = %f, current batch size = %d, learning_rate * (grad_weights_accumulators[i][j * mlp->layers_sizes[i-1] + k] / current_batch_size) = %f\n", learning_rate, grad_weights_accumulators[i][j * mlp->layers_sizes[i-1] + k], current_batch_size, learning_rate * (grad_weights_accumulators[i][j * mlp->layers_sizes[i-1] + k] / current_batch_size));
                    }

                }
                //next layer
            }
            //epoch_loss += batch_loss;
        }
        // all batches of this epoch  computed
        epoch_loss = epoch_loss/(train_dataset.size);
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
    printMLP(mlp);
    // Define learning parameters
    double learning_rate = 0.01;
    int num_epochs = 500;
    int batch_size = 1000; // Adjust based on your dataset size and memory constraints
    if (batch_size < NUM_THREADS){
        printf("Impossible to start the program, batch_size[%d] < num_thread[%d]", batch_size, NUM_THREADS);
        return 1;
    }
    // Train MLP
    trainMLP(splitted_dataset.train, mlp, num_epochs, batch_size, learning_rate);
    
    double error = evaluateMLP(mlp, splitted_dataset.test, sigmoid);
    printf("error is %f\n",error);

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

