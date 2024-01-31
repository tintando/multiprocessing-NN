#include <stdlib.h>
#include <math.h>

typedef struct MLP {
    int input_size;
    int output_size;
    int num_hidden_layers;
    int *hidden_layers_size; 
    double **neuron_activations;
    double **weights;   //weights 6 neuroni 2   layer[neurone*weight]       weights[layer][]
    double **biases;
} MLP;

typedef double (*ActivationFunction)(double);
typedef double (*ActivationFunctionDerivative)(double);

MLP *createMLP(int input_size, int output_size, int num_hidden_layers, int *hidden_layers_size);
void initializeXavier(double *weights, int in, int out);


MLP *createMLP(int input_size, int output_size, int num_hidden_layers, int *hidden_layers_size) {
    // Step 1: Allocate memory for the MLP structure itself
    MLP *mlp = (MLP *)malloc(sizeof(MLP));
    if (!mlp) return NULL;  // Check if allocation was successful

    // Step 2: Initialize basic properties
    mlp->input_size = input_size;
    mlp->output_size = output_size;
    mlp->num_hidden_layers = num_hidden_layers;

    // Step 3: Allocate memory for the array holding sizes of each hidden layer
    mlp->hidden_layers_size = (int *)malloc(num_hidden_layers * sizeof(int));
    if (!mlp->hidden_layers_size) {
        free(mlp);  // Free the MLP structure if allocation fails
        return NULL;
    }
    for (int i = 0; i < num_hidden_layers; i++) {
        mlp->hidden_layers_size[i] = hidden_layers_size[i];
    }

    // Step 4: Allocate memory for neuron activations, weights, and biases
    // Note: +1 in allocation size to include output layer
    mlp->neuron_activations = (double **)malloc((num_hidden_layers + 1) * sizeof(double *));
    mlp->weights = (double **)malloc((num_hidden_layers + 1) * sizeof(double *));
    mlp->biases = (double **)malloc((num_hidden_layers + 1) * sizeof(double *));
    if (!mlp->neuron_activations || !mlp->weights || !mlp->biases) {
        // Clean up in case of allocation failure
        free(mlp->hidden_layers_size);
        if (mlp->neuron_activations) free(mlp->neuron_activations);
        if (mlp->weights) free(mlp->weights);
        if (mlp->biases) free(mlp->biases);
        free(mlp);
        return NULL;
    }

    /*To access a specific weight that connects neuron j in layer i to neuron k in layer i+1, you would access weights[i][k * size_of_layer_i + j].
     Here, size_of_layer_i is either input_size for i = 0 or hidden_layers_size[i-1] for hidden layers.*/

    int prev_layer_size = input_size;  // Size of the previous layer; starts with input size
    for (int i = 0; i <= num_hidden_layers; i++) {
        // Determine the size of the current layer
        int layer_size = (i == num_hidden_layers) ? output_size : hidden_layers_size[i];

        // Allocate memory for neuron activations, weights, and biases of the current layer
        mlp->neuron_activations[i] = (double *)calloc(layer_size, sizeof(double));  // Initialized to 0
        mlp->weights[i] = (double *)malloc(prev_layer_size * layer_size * sizeof(double));
        mlp->biases[i] = (double *)calloc(layer_size, sizeof(double));  // Initialized to 0

        // Check for allocation failures
        if (!mlp->neuron_activations[i] || !mlp->weights[i] || !mlp->biases[i]) {
            // Free previously allocated memory in case of failure
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

        // Initialize weights here (e.g., using Xavier initialization)
        
        initializeXavier(mlp->weights[i], prev_layer_size, layer_size);

        prev_layer_size = layer_size;  // Update the size of the previous layer
    }

    return mlp;
}



void matrixMultiplyAndAddBias(double *output, double *input, double *weights, double *biases, int inputSize, int outputSize) {
    // For each neuron in the output layer
    for (int i = 0; i < outputSize; i++) {
        output[i] = 0.0; // Initialize output neuron activation 
        // Perform weighted sum of inputs for this neuron
        for (int j = 0; j < inputSize; j++) {
            output[i] += input[j] * weights[i * inputSize + j]; 
        }
        // Add bias for this neuron
        output[i] += biases[i];
    }
}



void initializeXavier(double *weights, int in, int out) {
    double limit = sqrt(6.0 / (in + out));  // Calculate the limit for uniform distribution
    for (int i = 0; i < in * out; i++) {
        weights[i] = (rand() / (RAND_MAX + 1.0)) * 2 * limit - limit; // Uniform distribution between -limit and limit
    }
}

void applyActivationFunction(double *layer, int size, double (*activationFunc)(double)) {
    for (int i = 0; i < size; i++) {
        layer[i] = activationFunc(layer[i]);
    }
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

double tanh(double x) {
    return tanh(x);
}

double dtanh(double x) {
    double tanh_x = tanh(x);
    return 1 - tanh_x * tanh_x;
}



void feedforward(MLP *mlp, double *input, ActivationFunction act) {
    // Copy input to the first layer's activations
    for (int i = 0; i < mlp->input_size; i++) {
        mlp->neuron_activations[0][i] = input[i];
    }

    // Process each layer
    for (int i = 0; i < mlp->num_hidden_layers; i++) {
        //zl=w*x+b
        matrixMultiplyAndAddBias(mlp->neuron_activations[i + 1], mlp->neuron_activations[i], mlp->weights[i], mlp->biases[i], i == 0 ? mlp->input_size : mlp->hidden_layers_size[i - 1], mlp->hidden_layers_size[i]);
        //al=activation(zl)
        applyActivationFunction(mlp->neuron_activations[i + 1], mlp->hidden_layers_size[i], act);
    }

    // Process the output layer
    matrixMultiplyAndAddBias(mlp->neuron_activations[mlp->num_hidden_layers], mlp->neuron_activations[mlp->num_hidden_layers - 1], mlp->weights[mlp->num_hidden_layers - 1], mlp->biases[mlp->num_hidden_layers - 1], mlp->hidden_layers_size[mlp->num_hidden_layers - 1], mlp->output_size);
    applyActivationFunction(mlp->neuron_activations[mlp->num_hidden_layers], mlp->output_size, act);
}


void backpropagation(MLP *mlp, double *input, double *target, ActivationFunction act, ActivationFunctionDerivative dact, double learning_rate, int batch_size) {
    // Forward pass 
    feedforward(mlp, input, act);

    // Allocate memory for storing gradients
    double **delta = (double **)malloc((mlp->num_hidden_layers + 1) * sizeof(double *));
    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        int layer_size = (i == mlp->num_hidden_layers) ? mlp->output_size : mlp->hidden_layers_size[i];
        delta[i] = (double *)calloc(layer_size, sizeof(double));
    }

    // Step 1: Calculate output layer error (delta)
    for (int i = 0; i < mlp->output_size; i++) {
        double error = target[i] - mlp->neuron_activations[mlp->num_hidden_layers][i];
        delta[mlp->num_hidden_layers][i] = error * dact(mlp->neuron_activations[mlp->num_hidden_layers][i]);
    }

    // Step 2: Propagate errors back through the network
    for (int i = mlp->num_hidden_layers - 1; i >= 0; i--) {
        int layer_size = i == 0 ? mlp->input_size : mlp->hidden_layers_size[i - 1];
        for (int j = 0; j < mlp->hidden_layers_size[i]; j++) {
            double error = 0.0;
            for (int k = 0; k < (i == mlp->num_hidden_layers - 1 ? mlp->output_size : mlp->hidden_layers_size[i + 1]); k++) {
                error += delta[i + 1][k] * mlp->weights[i][k * mlp->hidden_layers_size[i] + j];
            }
            delta[i][j] = error * dact(mlp->neuron_activations[i][j]);
        }
    }

    // Step 3: Update weights and biases for each layer
    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        int in_size = (i == 0) ? mlp->input_size : mlp->hidden_layers_size[i - 1];
        int out_size = (i == mlp->num_hidden_layers) ? mlp->output_size : mlp->hidden_layers_size[i];
        for (int j = 0; j < out_size; j++) {
            for (int k = 0; k < in_size; k++) {
                double grad_weight = delta[i][j] * (i == 0 ? input[k] : mlp->neuron_activations[i - 1][k]);
                mlp->weights[i][j * in_size + k] += learning_rate * grad_weight / batch_size;
            }
            mlp->biases[i][j] += learning_rate * delta[i][j] / batch_size;
        }
    }

    // Free allocated memory for delta
    for (int i = 0; i <= mlp->num_hidden_layers; i++) {
        free(delta[i]);
    }
    free(delta);
}

//dataset[sample][features]

void trainMLP(MLP *mlp, double **dataset, double **targets, int num_samples, int num_epochs, double learning_rate, int batch_size, ActivationFunction act, ActivationFunctionDerivative dact) {
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double total_loss = 0.0;

        // Loop over the dataset in batches
        for (int i = 0; i < num_samples; i += batch_size) {
            int current_batch_size = (i + batch_size > num_samples) ? (num_samples - i) : batch_size;

            // Process each sample in the batch
            for (int j = i; j < i + current_batch_size; j++) {
                // Backpropagation step
                backpropagation(mlp, dataset[j], targets[j], act, dact, learning_rate, current_batch_size);

                // Optional: Compute loss (e.g., mean squared error) for monitoring
                double loss = 0.0;
                for (int k = 0; k < mlp->output_size; k++) {
                    double error = targets[j][k] - mlp->neuron_activations[mlp->num_hidden_layers][k];
                    loss += error * error;
                }
                total_loss += loss / mlp->output_size;
            }
        }

        // Monitor the training progress
        total_loss /= num_samples;
        printf("Epoch %d, Loss: %f\n", epoch + 1, total_loss);
    }
}
