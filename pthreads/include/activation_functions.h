// neural_network.h
#ifndef activation_functions_h
#define activation_functions_h

#include <math.h>

// Activation functions and their derivatives
typedef double (*ActivationFunction)(double);
typedef double (*ActivationFunctionDerivative)(double);

// Activation functions
double sigmoid(double x);
double dsigmoid(double x);
double relu(double x);
double drelu(double x);
double dtanh(double x);

// Utility function for applying an activation function to a layer
void applyActivationFunction(double *logits, double *activations, int size, ActivationFunction activationFunc);

#endif // NEURAL_NETWORK_H
