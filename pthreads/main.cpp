#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "include/data_loading.h"
#include <string.h>
#include <pthread.h>

#define output_layer 1
#define previous_layer current_layer-1
#define next_layer current_layer+1
#define N_FEATURES 8
#define N_LABELS 1

/*structure of MLP
input size = # of nodes in first layer
output size = # of nodes in last layer
num_hidden_layers = # of hidden layers in neural network
hideen_layers_size = array where each entry is the number of nodes in that layer
ex: layer1 = 3 nodes, layer2 = 6, layer3 = 10 then hidden_layer_size=[3,6,10] and num_hidden_layers=3
neuron_activation[layer][node] = ...
weights[layer][hideen_layers_size[layer]*hideen_layers_size[layer-1]] =
               = weight between a node of the current layer and a node of previous layer
2d array biases[layer][node]*/
typedef struct MLP {
    int input_size; // number of nodes in first layer
    int output_size; // numebr of nodes in last layer
    int num_hidden_layers; // numebr of hidden layers in neural network
    int *hidden_layers_size; // array where each entry is the # of nodes in that layer
    double **neuron_activations;
    // weight between a node of the current layer and a node of previous layer 
    // [layer][hideen_layers_size[layer]*hideen_layers_size[layer-1]]
    double **weights; 
    double **biases;
} MLP;

typedef struct ThreadData {
    MLP *mlp;
    double **inputs;
    double **targets;
    int start_idx;
    int end_idx;
    ActivationFunction act;
    ActivationFunctionDerivative dact;
    double learning_rate;
    double ***grad_weights_accumulators;
    double **grad_biases_accumulator;
} ThreadData;   

//pointers for activation functions: Relu, sigmoid, tahn
typedef double (*ActivationFunction)(double);
//pointers for derivative of activation functions: dRelu, dsigmoid, dtahn
typedef double (*ActivationFunctionDerivative)(double);

//initializing all the functions for the compiler :)

MLP *createMLP(int input_size, int output_size, int num_hidden_layers, int *hidden_layers_size);
void initializeXavier(double *weights, int in, int out);
void feedforward(MLP *mlp, double *input, ActivationFunction act);
double backpropagation(MLP *mlp, double **inputs, double **targets, int current_batch_size, ActivationFunction act, ActivationFunctionDerivative dact, double learning_rate);
void trainMLP(MLP *mlp, double **dataset, double **targets, int num_samples, int num_epochs, double learning_rate, int batch_size, ActivationFunction act, ActivationFunctionDerivative dact);
double evaluateMLP(MLP *mlp, double **test_data, double **test_targets, int test_size, ActivationFunction act);
double sigmoid(double x);
double dsigmoid(double x);
double relu(double x);
double drelu(double x);
//double tanh(double x) already in math.h
double dtanh(double x);
void matrixMultiplyAndAddBias(double *output, double *input, double *weights, double *biases, int inputSize, int outputSize);
void applyActivationFunction(double *layer, int size, ActivationFunction activationFunc);
void initializeXavier(double *weights, int in, int out);
void loadAndPrepareDataset(const char* filename, double ***dataset, double ***targets, int *n_samples);
void shuffleDataset(double ***dataset, double ***targets, int n_samples);
void *threadTrainFunc(void *thread_data);
void aggregateAndApplyGradients(MLP *mlp, ThreadData *tdata, int num_threads, int batch_size, int input_size);


// Allocates memory for the MLP structures and initializes them,
// including the Xavier method for initializing weights.
MLP *createMLP(int input_size, int output_size, int num_hidden_layers, int *hidden_layers_size) {
    MLP *mlp = (MLP *)malloc(sizeof(MLP)); //mlp is the pointer to the MLP instance
    if (!mlp) return NULL;//in case of memory errors, this function returns NULL
    mlp->input_size = input_size;
    mlp->output_size = output_size;
    mlp->num_hidden_layers = num_hidden_layers;
    // allocate the array of sizes
    mlp->hidden_layers_size = (int *)malloc(num_hidden_layers * sizeof(int));
    if (!mlp->hidden_layers_size) {
        free(mlp);
        return NULL;
    }
    //initialize the array of sizes
    for (int i = 0; i < num_hidden_layers; i++) {
        mlp->hidden_layers_size[i] = hidden_layers_size[i];
    }
    // allocate neuron_activations, weights, biases
    mlp->neuron_activations = (double **)malloc((num_hidden_layers + 1) * sizeof(double *));
    mlp->weights = (double **)malloc((num_hidden_layers + 1) * sizeof(double *));
    mlp->biases = (double **)malloc((num_hidden_layers + 1) * sizeof(double *));
    if (!mlp->neuron_activations || !mlp->weights || !mlp->biases) {
        //errors
        free(mlp->hidden_layers_size);
        if (mlp->neuron_activations) free(mlp->neuron_activations);
        if (mlp->weights) free(mlp->weights);
        if (mlp->biases) free(mlp->biases);
        free(mlp);
        return NULL;
    }
    // initialize neuron_activations, weights, biases
    int prev_layer_size = input_size;//keeps track of the size of previous layer, starts with size of input
    for (int i = 0; i <= num_hidden_layers; i++) {
        int layer_size = (i == num_hidden_layers) ? output_size : hidden_layers_size[i]; //size of current layer
        mlp->neuron_activations[i] = (double *)calloc(layer_size, sizeof(double)); // allocates and initializes to 0
        mlp->weights[i] = (double *)malloc(prev_layer_size * layer_size * sizeof(double)); //allocates
        mlp->biases[i] = (double *)calloc(layer_size, sizeof(double)); // allocates and initializes to 0

        if (!mlp->neuron_activations[i] || !mlp->weights[i] || !mlp->biases[i]) {
            for (int j = 0; j < i; j++) {
                free(mlp->neuron_activations[j]);
                free(mlp->weights[j]);
                free(mlp->biases[j]);
            }
            free(mlp->neuron_activations);
            free(mlp->weights);
            free(mlp->biases);
            free(mlp->hidden_layers_size);
            free(mlp);
            return NULL;
        }

        initializeXavier(mlp->weights[i], prev_layer_size, layer_size); //initialize weights

        prev_layer_size = layer_size;
    }

    return mlp;
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
/* mlp = Pointer to the MLP structure to be trained.
   input = A sample (array of features)
   act = Activation function used in the neurons during forward propagation.
   Propagates the input through the network to produce an output.
   It computes the activations of all neurons in the network using the input data, weights, and biases.*/
void feedforward(MLP *mlp, double *input, ActivationFunction act) {
    // gives input
    for (int i = 0; i < mlp->input_size; i++) {
        //Initialize the activation of the input layer neurons with the input values.
        mlp->neuron_activations[0][i] = input[i];
    }

    // compute zl and al
    for (int current_layer = 0; current_layer < mlp->num_hidden_layers; current_layer++) { //for each hidden layer
        matrixMultiplyAndAddBias(mlp->neuron_activations[next_layer],
                                mlp->neuron_activations[current_layer],
                                mlp->weights[current_layer], mlp->biases[current_layer],
                                current_layer == 0 ? mlp->input_size : mlp->hidden_layers_size[previous_layer],
                                mlp->hidden_layers_size[current_layer]
                                );
        applyActivationFunction(mlp->neuron_activations[next_layer], mlp->hidden_layers_size[current_layer], act);
    }
    //same process for output layer
    matrixMultiplyAndAddBias(mlp->neuron_activations[mlp->num_hidden_layers],
                             mlp->neuron_activations[mlp->num_hidden_layers - 1],
                             mlp->weights[mlp->num_hidden_layers - 1],
                             mlp->biases[mlp->num_hidden_layers - 1],
                             mlp->hidden_layers_size[mlp->num_hidden_layers - 1], 
                             mlp->output_size);
    applyActivationFunction(mlp->neuron_activations[mlp->num_hidden_layers], mlp->output_size, act);
}

/* mlp = Pointer to the MLP structure to be trained.
   inputs = batch part of the two-dimensional array of inputs, each row represents a sample.
   targets = batch part of the two-dimensional array of target outputs
   current_batch_size = the size of this batch
   act = Activation function used in the neurons during forward propagation.
   dact = Derivative of the activation function used during backpropagation.
   learning_rate = The step size at each iteration while moving toward a minimum of the loss function.
   Adjusts the weights and biases to minimize the error between the actual output and the predicted output by the network. 
   This function calculates gradients for weights and biases using the chain rule
   and updates them accordingly.*/

double backpropagation(MLP *mlp, double **inputs, double **targets, int current_batch_size, ActivationFunction act, ActivationFunctionDerivative dact, double learning_rate) {
    // Initialize gradient accumulators for weights and biases to zero
    double ***grad_weights_accumulators = (double ***)malloc((mlp->num_hidden_layers + output_layer) * sizeof(double **));
    double **grad_biases_accumulator = (double **)malloc((mlp->num_hidden_layers + output_layer) * sizeof(double *));
    
    for (int current_layer = 0; current_layer <= mlp->num_hidden_layers; current_layer++) {
        int size_in = (current_layer == 0) ? mlp->input_size : mlp->hidden_layers_size[previous_layer];
        int size_out = (current_layer == mlp->num_hidden_layers) ? mlp->output_size : mlp->hidden_layers_size[current_layer];
        grad_weights_accumulators[current_layer] = (double **)malloc(size_out * sizeof(double *));
        grad_biases_accumulator[current_layer] = (double *)calloc(size_out, sizeof(double));
        for (int neuron = 0; neuron < size_out; neuron++) {
            grad_weights_accumulators[current_layer][neuron] = (double *)calloc(size_in, sizeof(double));
        }
    }

    double **delta = (double **)malloc((mlp->num_hidden_layers + output_layer) * sizeof(double *));
    for (int layer = 0; layer <= mlp->num_hidden_layers; layer++) {
        int layer_size = (layer == mlp->num_hidden_layers) ? mlp->output_size : mlp->hidden_layers_size[layer];
        delta[layer] = (double *)malloc(layer_size * sizeof(double));
    }

    double batch_loss = 0.0;

    for (int sample = 0; sample < current_batch_size; sample++) {
        feedforward(mlp, inputs[sample], act);
        double sample_loss=0.0;
        for (int i = 0; i < mlp->output_size; i++) {
            double output_error = targets[sample][i] - mlp->neuron_activations[mlp->num_hidden_layers][i];
            delta[mlp->num_hidden_layers][i] = output_error * dact(mlp->neuron_activations[mlp->num_hidden_layers][i]);
            sample_loss+=output_error*output_error;
        }
        batch_loss+=sample_loss;

        for (int current_layer = mlp->num_hidden_layers - 1; current_layer >= 0; current_layer--) {
            int size_out = (current_layer == mlp->num_hidden_layers) ? mlp->output_size : mlp->hidden_layers_size[current_layer];
            int size_in = (current_layer == 0) ? mlp->input_size : mlp->hidden_layers_size[previous_layer];
            
            for (int j = 0; j < size_out; j++) {
                double error = 0.0;
                for (int k = 0; k < ((current_layer == mlp->num_hidden_layers - 1) ? mlp->output_size : mlp->hidden_layers_size[next_layer]); k++) {
                    error += mlp->weights[next_layer][k * size_out + j] * delta[next_layer][k];
                }
                delta[current_layer][j] = error * dact(mlp->neuron_activations[current_layer][j]);
            }

            for (int neuron = 0; neuron < size_out; neuron++) {
                for (int input_neuron = 0; input_neuron < size_in; input_neuron++) {
                    double grad = delta[current_layer][neuron] * (current_layer == 0 ? inputs[sample][input_neuron] : mlp->neuron_activations[previous_layer][input_neuron]);
                    grad_weights_accumulators[current_layer][neuron][input_neuron] += grad;
                }
                grad_biases_accumulator[current_layer][neuron] += delta[current_layer][neuron];
            }
        }
    }

    for (int current_layer = 0; current_layer <= mlp->num_hidden_layers; current_layer++) {
        int size_in = (current_layer == 0) ? mlp->input_size : mlp->hidden_layers_size[previous_layer];
        int size_out = (current_layer == mlp->num_hidden_layers) ? mlp->output_size : mlp->hidden_layers_size[current_layer];
        
        for (int neuron = 0; neuron < size_out; neuron++) {
            for (int input_neuron = 0; input_neuron < size_in; input_neuron++) {
                double mean_grad = grad_weights_accumulators[current_layer][neuron][input_neuron] / current_batch_size;
                mlp->weights[current_layer][neuron * size_in + input_neuron] += learning_rate * mean_grad;
            }
            double mean_grad_bias = grad_biases_accumulator[current_layer][neuron] / current_batch_size;
            mlp->biases[current_layer][neuron] += learning_rate * mean_grad_bias;
        }
    }

    for (int layer = 0; layer <= mlp->num_hidden_layers; layer++) {
        for (int neuron = 0; neuron < (layer == mlp->num_hidden_layers ? mlp->output_size : mlp->hidden_layers_size[layer]); neuron++) {
            free(grad_weights_accumulators[layer][neuron]);
        }
        free(grad_weights_accumulators[layer]);
        free(grad_biases_accumulator[layer]);
        free(delta[layer]);
    }
    free(grad_weights_accumulators);
    free(grad_biases_accumulator);
    free(delta);
    return batch_loss / (current_batch_size * mlp->output_size);
}

/*  mlp = Pointer to the MLP structure to be trained.
    dataset = Two-dimensional array of input data, where each row represents a sample.
    targets = Two-dimensional array of target outputs corresponding to each input sample.
    num_samples = The total number of samples in the dataset.
    num_epochs = The number of times the entire dataset is passed through the network for training.
    learning_rate = The step size at each iteration while moving toward a minimum of the loss function.
    batch_size= The number of samples processed before the model is updated.
    act = Activation function used in the neurons during forward propagation.
    dact = Derivative of the activation function used during backpropagation.
   Repeatedly applies feedforward and backpropagation on the dataset for a specified number of epochs,
   adjusting the weights to minimize the loss.*/
void trainMLP(MLP *mlp, double **dataset, double **targets, int num_samples, int num_epochs, double learning_rate, int batch_size, ActivationFunction act, ActivationFunctionDerivative dact) {
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        //An epoch is a single pass through the entire dataset.
        shuffleDataset(&dataset, &targets, num_samples);
        double total_loss = 0.0; //accomulator of loss over a single epoch
        for (int i = 0; i < num_samples; i += batch_size) { // iterate through the dataset in batches.
            int current_batch_size = (i + batch_size > num_samples) ? (num_samples - i) : batch_size;// if it's the last batch, probably it's smaller than others
            double **batch_inputs = (double **)malloc(current_batch_size * sizeof(double *));// the inputs for this batch
            double **batch_targets = (double **)malloc(current_batch_size * sizeof(double *));// the labels for this batch
            for (int j = 0; j < current_batch_size; j++) {
                batch_inputs[j] = dataset[i + j];
                batch_targets[j] = targets[i + j];
            }
            double batch_loss = backpropagation(mlp, batch_inputs, batch_targets, current_batch_size, act, dact, learning_rate);
            free(batch_inputs);
            free(batch_targets);
            
            
            total_loss += batch_loss; //add to the total loss the average loss of this batch
        }
        //by printing the average loss of this epoch we have an idea of how good the learning is going odd
        total_loss /= (num_samples/batch_size); 
        printf("Epoch %d, Loss: %f\n", epoch + 1, total_loss);
    }
}

double evaluateMLP(MLP *mlp, double **test_data, double **test_targets, int test_size, ActivationFunction act) {
    double total_error = 0.0;
    for (int i = 0; i < test_size; i++) {
        // Feedforward to get the prediction
        feedforward(mlp, test_data[i], act);

        // Assuming the last layer's activations are the predictions
        double *prediction = mlp->neuron_activations[mlp->num_hidden_layers];
        
        // Calculate error for this sample
        for (int j = 0; j < mlp->output_size; j++) {
            double error = test_targets[i][j] - prediction[j];
            total_error += error * error; // For MSE, sum the squared error
        }
    }

    // Return average MSE over the test set
    return total_error / (test_size * mlp->output_size);
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double dsigmoid(double x) {
    double sigmoid_x = sigmoid(x);
    return sigmoid_x * (1 - sigmoid_x);
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double drelu(double x) {
    return x > 0 ? 1 : 0;
}


double dtanh(double x) {
    double tanh_x = tanh(x);
    return 1 - tanh_x * tanh_x;
}

/* output = array of wheighted sums (neurons of current layer = neuron_activations of next layer)
   input =  array of wheighted sums (neuron_activations of current layer = neurons of previous layer)
   weights = weights of current layer
   biases = biases of current layer
   input_size = number of neurons of previous layer
   outputsize = number of neurons of current layer
*/
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

void applyActivationFunction(double *layer, int size, ActivationFunction activationFunc) {
    for (int i = 0; i < size; i++) {
        layer[i] = activationFunc(layer[i]);
    }
}


void loadAndPrepareDataset(const char* filename, double ***dataset, double ***targets, int *n_samples) {
    // Read the dataset
    Sample* samples = readDataset(filename, n_samples);//returns a 1D array
    if (!samples || *n_samples <= 0) {
        printf("Error reading dataset or dataset is empty.\n");
        return;
    }

    // Allocate memory to split the array in labels and features
    float* h_features = (float*)malloc(*n_samples * N_FEATURES * sizeof(float));
    float* h_labels = (float*)malloc(*n_samples * N_LABELS * sizeof(float));
    if (!h_features || !h_labels) {
        printf("Failed to allocate memory for features and labels.\n");
        free(samples); // Assuming samples need to be freed here
        return;
    }

    // Copy data from samples to h_features and h_labels
    for (int i = 0; i < *n_samples; i++) {
        memcpy(h_features + i * N_FEATURES, samples[i].features, N_FEATURES * sizeof(float));
        h_labels[i] = samples[i].label;
    }

    // Assuming the responsibility to free samples is here
    // Remember to free the samples' features if they're dynamically allocated
    free(samples);

    // Allocate memory for dataset and targets
    *dataset = (double**) malloc(*n_samples * sizeof(double*));
    *targets = (double**) malloc(*n_samples * sizeof(double*));
    for (int i = 0; i < *n_samples; i++) {
        (*dataset)[i] = (double*) malloc(N_FEATURES * sizeof(double));
        (*targets)[i] = (double*) malloc(N_LABELS * sizeof(double));
        for (int j = 0; j < N_FEATURES; j++) {
            (*dataset)[i][j] = (double)h_features[i * N_FEATURES + j];
        }
        (*targets)[i][0] = (double)h_labels[i];
    }

    // Free temporary host memory
    free(h_features);
    free(h_labels);
}

void shuffleDataset(double ***dataset, double ***targets, int n_samples) {
    srand(time(NULL)); // Seed the random number generator with current time

    for (int i = 0; i < n_samples - 1; i++) {
        int j = i + rand() / (RAND_MAX / (n_samples - i) + 1); // Generate a random index from i to n_samples-1

        // Swap dataset[i] and dataset[j]
        double *temp_dataset = (*dataset)[i];
        (*dataset)[i] = (*dataset)[j];
        (*dataset)[j] = temp_dataset;

        // Swap targets[i] and targets[j] similarly
        double *temp_targets = (*targets)[i];
        (*targets)[i] = (*targets)[j];
        (*targets)[j] = temp_targets;
    }
}


//splits the dataset in train, validation and test set
void splitDataset(int *train_size, int *test_size, int *validation_size, 
                double*** train_data, double*** train_targets, 
                double*** test_data, double*** test_targets, 
                double*** validation_data, double*** validation_targets, 
                double*** dataset, double*** targets, int *n_samples){
    *train_data = (double**) malloc(*train_size * sizeof(double*));
    *train_targets = (double**) malloc(*train_size * sizeof(double*));
    for (int i = 0; i < *train_size; i++) {
        (*train_data)[i] = (double*) malloc(N_FEATURES * sizeof(double));
        (*train_targets)[i] = (double*) malloc(N_LABELS * sizeof(double));
        for (int j=0; j<N_FEATURES; j++){
            (*train_data)[i][j] = (*dataset)[i][j];
        }
        (*train_targets)[i][0] = (*targets)[i][0];
    }
    *test_data = (double**) malloc(*test_size * sizeof(double*));
    *test_targets = (double**) malloc(*test_size * sizeof(double*));
    for (int i = 0; i < *test_size; i++) {
        (*test_data)[i] = (double*) malloc(N_FEATURES * sizeof(double));
        (*test_targets)[i] = (double*) malloc(N_LABELS * sizeof(double));
        for (int j=0; j<N_FEATURES; j++){
            (*test_data)[i][j] = (*dataset)[i + *train_size][j];
        }
        (*test_targets)[i][0] = (*targets)[i + *train_size][0];
    }
    *validation_data = (double**) malloc(*validation_size * sizeof(double*));
    *validation_targets = (double**) malloc(*validation_size * sizeof(double*));
    for (int i = 0; i < *validation_size; i++) {
        (*validation_data)[i] = (double*) malloc(N_FEATURES * sizeof(double));
        (*validation_targets)[i] = (double*) malloc(N_LABELS * sizeof(double));
        for (int j=0; j<N_FEATURES; j++){
            (*validation_data)[i][j] = (*dataset)[i + *train_size + *test_size][j];
        }
        (*test_targets)[i][0] = (*targets)[i+ *train_size + *validation_size][0];
    }
}


void trainMLPWithPthreads(MLP *mlp, double **dataset, double **targets, int num_samples, int num_epochs, double learning_rate, int batch_size, ActivationFunction act, ActivationFunctionDerivative dact, int num_threads) {
    pthread_t threads[num_threads];
    ThreadData tdata[num_threads];

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        // Shuffle dataset at the beginning of each epoch

        for (int batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
            int current_batch_size = batch_start + batch_size > num_samples ? num_samples - batch_start : batch_size;

            // Initialize thread data for each thread
            for (int i = 0; i < num_threads; ++i) {
                int start_idx = i * (current_batch_size / num_threads);
                int end_idx = (i + 1) * (current_batch_size / num_threads) - 1;
                if (i == num_threads - 1) {
                    // Ensure the last thread picks up any remaining samples due to integer division
                    end_idx = current_batch_size - 1;
                }

                // Adjust start and end index based on the batch_start
                start_idx += batch_start;
                end_idx += batch_start;

                tdata[i].mlp = mlp;
                tdata[i].inputs = dataset + start_idx;
                tdata[i].targets = targets + start_idx;
                tdata[i].start_idx = start_idx;
                tdata[i].end_idx = end_idx;
                tdata[i].act = act;
                tdata[i].dact = dact;
                tdata[i].learning_rate = learning_rate;

                // Allocate memory for gradients; to be implemented by the user
                // Example: tdata[i].grad_weights_accumulators = allocateGradientStorage(...);
                // Example: tdata[i].grad_biases_accumulator = allocateGradientStorage(...);

                pthread_create(&threads[i], NULL, threadTrainFunc, (void*)&tdata[i]);
            }

            // Join threads and aggregate gradients
            for (int i = 0; i < num_threads; ++i) {
                pthread_join(threads[i], NULL);
            }

            // Aggregate gradients and update weights; to be implemented by the user
            // Example: aggregateAndApplyGradients(mlp, tdata, num_threads, current_batch_size, mlp->input_size);
        }
        // Optionally, print epoch and loss information here
    }

    // Cleanup allocations for gradients; to be implemented by the user
}


void *threadTrainFunc(void *thread_data) {
    ThreadData *data = (ThreadData*) thread_data;
    // Implement the feedforward and backpropagation for the subset of data
    // Store computed gradients in thread-local structures
    // Remember to correctly index into the global dataset based on start_idx and end_idx
    // Example: feedforward and backpropagation logic here

    // The actual implementation would involve adjusting the existing backpropagation
    // function or creating a new one that can handle partial datasets and store gradients locally

    pthread_exit(NULL);
}

void aggregateAndApplyGradients(MLP *mlp, ThreadData *tdata, int num_threads, int batch_size, int input_size) {
    // This function should aggregate gradients from all threads and then update the global weights and biases
    // Ensure this operation is thread-safe if using threads outside of the main training loop
    // Example: Loop over all threads, aggregate their stored gradients, then update MLP weights and biases accordingly
}


int main(int argc, char *argv[]){

    const char* filename = "/home/pavka/multiprocessing-NN/serial/datasets/california.csv";
    double **dataset = NULL, **targets = NULL;
    int n_samples = 0;

    // Load and prepare the dataset
    loadAndPrepareDataset(filename, &dataset, &targets, &n_samples);
    double **train_data = NULL, **train_targets = NULL, **test_data = NULL,
            **test_targets = NULL, **validation_data = NULL, **validation_targets = NULL;
    int train_size = (int)(n_samples*80/100), test_size = (int)(n_samples*20/100), validation_size = 0;
    if(train_size+test_size!=n_samples){
        return 1;
    }
    splitDataset(&train_size, &test_size, &validation_size, &train_data, &train_targets, &test_data, &test_targets, &validation_data, &validation_targets, &dataset, &targets, &n_samples);
    // Initialize your MLP
    int input_size = N_FEATURES; // Define according to your dataset
    int output_size = N_LABELS; // Typically 1 for regression tasks
    int num_hidden_layers = 2; // Example: 2 hidden layers
    int hidden_layers_size[] = {5, 2}; // Example sizes for the hidden layers
    MLP *mlp = createMLP(input_size, output_size, num_hidden_layers, hidden_layers_size);

    // Define learning parameters
    double learning_rate = 0.01;
    int num_epochs = 500;
    int batch_size = 64; // Adjust based on your dataset size and memory constraints
    // Train MLP
    trainMLP(mlp, train_data, train_targets, train_size, num_epochs, learning_rate, batch_size, sigmoid, dsigmoid);
    double error = evaluateMLP(mlp,test_data,test_targets,test_size, sigmoid);
    printf("error is %f\n",error);

    // Clean up
    for (int i = 0; i < n_samples; i++) {
        free(dataset[i]);
        free(targets[i]);
    }
    free(dataset);
    free(targets);

    return 0;
}