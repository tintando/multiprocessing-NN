#include <math.h>

//pointers for activation functions: Relu, sigmoid, tahn
typedef double (*ActivationFunction)(double);
//pointers for derivative of activation functions: dRelu, dsigmoid, dtahn
typedef double (*ActivationFunctionDerivative)(double);

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

void applyActivationFunction(double *logits, double *activations, int size, ActivationFunction activationFunc) {
    for (int i = 0; i < size; i++) {
        activations[i] = activationFunc(logits[i]);
    }
}