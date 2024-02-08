#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*structure of MLP
input and output size are the number of nodes in the first and last layer
hideen_layers_size is an array that contains the number of nodes in each layer, while num_hidden layer is the len of this array,
ex. layer1=3, layer2=6, layer3=10,layer4=2      then hidden_layer_size=[3,6,10,2] and num_hidden_layers=4
2d array neuron_activation[layer][node]
2d array weights[layer][i * num_of_input_node + j]  , the weights of a neuron at layer L and index I is weights[L][I*num_of_input_node + j], with 0<j<num_of_input_node
2d array biases[layer][node]
*/
typedef struct MLP {
    int input_size;
    int output_size;
    int num_hidden_layers;
    int *hidden_layers_size;
    double **neuron_activations;
    double **weights;
    double **biases;
} MLP;

//pointers for activation functions(Relu,sigmoid,tahn)
typedef double (*ActivationFunction)(double);
typedef double (*ActivationFunctionDerivative)(double);

//initialize all the functions
MLP *createMLP(int input_size, int output_size, int num_hidden_layers, int *hidden_layers_size);
void initializeXavier(double *weights, int in, int out);
void feedforward(MLP *mlp, double *input, ActivationFunction act);
void backpropagation(MLP *mlp, double **inputs, double **targets, int current_batch_size, ActivationFunction act, ActivationFunctionDerivative dact, double learning_rate);
void trainMLP(MLP *mlp, double **dataset, double **targets, int num_samples, int num_epochs, double learning_rate, int batch_size, ActivationFunction act, ActivationFunctionDerivative dact);
double sigmoid(double x);
double dsigmoid(double x);
double relu(double x);
double drelu(double x);
double tanh(double x);
double dtanh(double x);
void matrixMultiplyAndAddBias(double *output, double *input, double *weights, double *biases, int inputSize, int outputSize);
void applyActivationFunction(double *layer, int size, ActivationFunction activationFunc);
void initializeXavier(double *weights, int in, int out);


//constructor for MLP
MLP *createMLP(int input_size, int output_size, int num_hidden_layers, int *hidden_layers_size) {
    MLP *mlp = (MLP *)malloc(sizeof(MLP));
    if (!mlp) return NULL;

    mlp->input_size = input_size;
    mlp->output_size = output_size;
    mlp->num_hidden_layers = num_hidden_layers;

    mlp->hidden_layers_size = (int *)malloc(num_hidden_layers * sizeof(int));
    if (!mlp->hidden_layers_size) {
        free(mlp);
        return NULL;
    }
    for (int i = 0; i < num_hidden_layers; i++) {
        mlp->hidden_layers_size[i] = hidden_layers_size[i];
    }

    mlp->neuron_activations = (double **)malloc((num_hidden_layers + 1) * sizeof(double *));
    mlp->weights = (double **)malloc((num_hidden_layers + 1) * sizeof(double *));
    mlp->biases = (double **)malloc((num_hidden_layers + 1) * sizeof(double *));
    if (!mlp->neuron_activations || !mlp->weights || !mlp->biases) {
        free(mlp->hidden_layers_size);
        if (mlp->neuron_activations) free(mlp->neuron_activations);
        if (mlp->weights) free(mlp->weights);
        if (mlp->biases) free(mlp->biases);
        free(mlp);
        return NULL;
    }

    int prev_layer_size = input_size;
    for (int i = 0; i <= num_hidden_layers; i++) {
        int layer_size = (i == num_hidden_layers) ? output_size : hidden_layers_size[i];

        mlp->neuron_activations[i] = (double *)calloc(layer_size, sizeof(double));
        mlp->weights[i] = (double *)malloc(prev_layer_size * layer_size * sizeof(double));
        mlp->biases[i] = (double *)calloc(layer_size, sizeof(double));

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

        initializeXavier(mlp->weights[i], prev_layer_size, layer_size);

        prev_layer_size = layer_size;
    }

    return mlp;
}

//popular way to initialize weights
void initializeXavier(double *weights, int in, int out) {
    double limit = sqrt(6.0 / (in + out));
    for (int i = 0; i < in * out; i++) {
        weights[i] = (rand() / (double)(RAND_MAX)) * 2 * limit - limit;
    }
}

//feedforward algorithm
void feedforward(MLP *mlp, double *input, ActivationFunction act) {
    //gives input
    for (int i = 0; i < mlp->input_size; i++) {
        mlp->neuron_activations[0][i] = input[i];
    }

    //compute zl and al
    for (int i = 0; i < mlp->num_hidden_layers; i++) {
        matrixMultiplyAndAddBias(mlp->neuron_activations[i + 1], mlp->neuron_activations[i], mlp->weights[i], mlp->biases[i], i == 0 ? mlp->input_size : mlp->hidden_layers_size[i - 1], mlp->hidden_layers_size[i]);
        applyActivationFunction(mlp->neuron_activations[i + 1], mlp->hidden_layers_size[i], act);
    }

    matrixMultiplyAndAddBias(mlp->neuron_activations[mlp->num_hidden_layers], mlp->neuron_activations[mlp->num_hidden_layers - 1], mlp->weights[mlp->num_hidden_layers - 1], mlp->biases[mlp->num_hidden_layers - 1], mlp->hidden_layers_size[mlp->num_hidden_layers - 1], mlp->output_size);
    applyActivationFunction(mlp->neuron_activations[mlp->num_hidden_layers], mlp->output_size, act);
}

void backpropagation(MLP *mlp, double **inputs, double **targets, int current_batch_size, ActivationFunction act, ActivationFunctionDerivative dact, double learning_rate) {
    // Initialize gradient accumulators for weights and biases to zero
    double ***grad_weights_accumulator = (double ***)malloc((mlp->num_hidden_layers + 1) * sizeof(double **));
    double **grad_biases_accumulator = (double **)malloc((mlp->num_hidden_layers + 1) * sizeof(double *));
    
    for (int layer = 0; layer <= mlp->num_hidden_layers; layer++) {
        //size input and output layer
        int size_in = (layer == 0) ? mlp->input_size : mlp->hidden_layers_size[layer - 1];
        int size_out = (layer == mlp->num_hidden_layers) ? mlp->output_size : mlp->hidden_layers_size[layer];

        grad_weights_accumulator[layer] = (double **)malloc(size_out * sizeof(double *));
        grad_biases_accumulator[layer] = (double *)calloc(size_out, sizeof(double));
        for (int neuron = 0; neuron < size_out; neuron++) {
            grad_weights_accumulator[layer][neuron] = (double *)calloc(size_in, sizeof(double));
        }
    }

    // Allocate memory for delta values
    double **delta = (double **)malloc((mlp->num_hidden_layers + 1) * sizeof(double *));
    for (int layer = 0; layer <= mlp->num_hidden_layers; layer++) {
        int layer_size = (layer == mlp->num_hidden_layers) ? mlp->output_size : mlp->hidden_layers_size[layer];
        delta[layer] = (double *)malloc(layer_size * sizeof(double));
    }

    // Process each sample in the batch
    for (int sample = 0; sample < current_batch_size; sample++) {
        // Forward pass
        feedforward(mlp, inputs[sample], act);

        // Calculate output layer delta
        for (int i = 0; i < mlp->output_size; i++) {
            double output_error = targets[sample][i] - mlp->neuron_activations[mlp->num_hidden_layers][i];
            delta[mlp->num_hidden_layers][i] = output_error * dact(mlp->neuron_activations[mlp->num_hidden_layers][i]);
        }

        // Backpropagate the error
        for (int layer = mlp->num_hidden_layers - 1; layer >= 0; layer--) {
            int size_out = (layer == mlp->num_hidden_layers) ? mlp->output_size : mlp->hidden_layers_size[layer];
            int size_in = (layer == 0) ? mlp->input_size : mlp->hidden_layers_size[layer - 1];
            
            for (int j = 0; j < size_out; j++) {
                double error = 0.0;
                for (int k = 0; k < ((layer == mlp->num_hidden_layers - 1) ? mlp->output_size : mlp->hidden_layers_size[layer + 1]); k++) {
                    error += mlp->weights[layer + 1][k * size_out + j] * delta[layer + 1][k];
                }
                delta[layer][j] = error * dact(mlp->neuron_activations[layer][j]);
            }

            // Accumulate gradients for weights and biases
            for (int neuron = 0; neuron < size_out; neuron++) {
                for (int input_neuron = 0; input_neuron < size_in; input_neuron++) {
                    double grad = delta[layer][neuron] * (layer == 0 ? inputs[sample][input_neuron] : mlp->neuron_activations[layer - 1][input_neuron]);
                    grad_weights_accumulator[layer][neuron][input_neuron] += grad;
                }
                grad_biases_accumulator[layer][neuron] += delta[layer][neuron];
            }
        }
    }

    // Apply mean gradients to update weights and biases
    for (int layer = 0; layer <= mlp->num_hidden_layers; layer++) {
        int size_in = (layer == 0) ? mlp->input_size : mlp->hidden_layers_size[layer - 1];
        int size_out = (layer == mlp->num_hidden_layers) ? mlp->output_size : mlp->hidden_layers_size[layer];
        
        for (int neuron = 0; neuron < size_out; neuron++) {
            for (int input_neuron = 0; input_neuron < size_in; input_neuron++) {
                // Calculate mean gradient
                double mean_grad = grad_weights_accumulator[layer][neuron][input_neuron] / current_batch_size;
                // Update weights
                mlp->weights[layer][neuron * size_in + input_neuron] += learning_rate * mean_grad;
            }
            // Calculate mean gradient for biases and update
            double mean_grad_bias = grad_biases_accumulator[layer][neuron] / current_batch_size;
            mlp->biases[layer][neuron] += learning_rate * mean_grad_bias;
        }
    }

    // Free memory allocated for gradient accumulators and delta
    for (int layer = 0; layer <= mlp->num_hidden_layers; layer++) {
        for (int neuron = 0; neuron < (layer == mlp->num_hidden_layers ? mlp->output_size : mlp->hidden_layers_size[layer]); neuron++) {
            free(grad_weights_accumulator[layer][neuron]);
        }
        free(grad_weights_accumulator[layer]);
        free(grad_biases_accumulator[layer]);
        free(delta[layer]);
    }
    free(grad_weights_accumulator);
    free(grad_biases_accumulator);
    free(delta);
}

void trainMLP(MLP *mlp, double **dataset, double **targets, int num_samples, int num_epochs, double learning_rate, int batch_size, ActivationFunction act, ActivationFunctionDerivative dact) {
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        double total_loss = 0.0;

        for (int i = 0; i < num_samples; i += batch_size) {
            int current_batch_size = (i + batch_size > num_samples) ? (num_samples - i) : batch_size;

            double **batch_inputs = (double **)malloc(current_batch_size * sizeof(double *));
            double **batch_targets = (double **)malloc(current_batch_size * sizeof(double *));
            for (int j = 0; j < current_batch_size; j++) {
                batch_inputs[j] = dataset[i + j];
                batch_targets[j] = targets[i + j];
            }

            backpropagation(mlp, batch_inputs, batch_targets, current_batch_size, act, dact, learning_rate);

            free(batch_inputs);
            free(batch_targets);

            double loss = 0.0;
            for (int j = i; j < i + current_batch_size; j++) {
                for (int k = 0; k < mlp->output_size; k++) {
                    double error = targets[j][k] - mlp->neuron_activations[mlp->num_hidden_layers][k];
                    loss += error * error;
                }
            }
            total_loss += loss / (mlp->output_size * current_batch_size);
        }

        total_loss /= num_samples;
        printf("Epoch %d, Loss: %f\n", epoch + 1, total_loss);
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

void matrixMultiplyAndAddBias(double *output, double *input, double *weights, double *biases, int inputSize, int outputSize) {
    for (int i = 0; i < outputSize; i++) {
        output[i] = 0.0;
        for (int j = 0; j < inputSize; j++) {
            output[i] += input[j] * weights[i * inputSize + j];
        }
        output[i] += biases[i];
    }
}

void applyActivationFunction(double *layer, int size, ActivationFunction activationFunc) {
    for (int i = 0; i < size; i++) {
        layer[i] = activationFunc(layer[i]);
    }
}
