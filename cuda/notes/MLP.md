# Parameters
eg. `[16, 8, 8, 4]`
- Input size
- Hidden layer sizes
- Output size

eg. `[RELU, ReLU, Sigmoid]`
- Activation functions

eg. `Cross entropy loss`
- Loss Function

eg. `SGD`
- Optimizer

![[usomemoria.png]]
Gradients are freed in zero grad
# Data loading
Load dataset from file
Load in batches, shuffled
# Forward
```
a_i = afunc(W_i*a_(i-1)+b_i)
|a_i| = i layer size
W_i_rows = i-1 layer size 
W_i_columns = i layer size
|b_i| = i layer size

eg.
input --- output
[8, 16, 8, 1]

W_0 = 8x16, b_0 = 16
W_1 = 16x8, b_1 = 8
W_2 = 8x1, b_2 = 1
```
Store all a_i and logit_i (before afunc)

# Backward
1. compute partial derivatives to outputs
	1. suppose `L = 1n/2n * Σ(y_i - t_i)^2 `and activation = sigmoid, the below are vector operations where dimension is last layer
	2. `gradient_a^L = (predicted-target)`
	3. `sigmoid'(logit) = sigmoid(logit)*(1-sigmoid(logit))`
	4. d^L = grad * sig' (hadamard)
2. obtain other d^l
	1. `d^l = d^(l+1)*W^l+1^T * afunc'(logit^l) (hadamard))`
		1. eg. 1x4 4x8 8x1
3. get final gradients
	1. g_b = d^l
	2. `g_w_jk = a_k^(l-1)*d_j^l` 
4. do this for every sample in batch, then average the gradients and multiply by learning rate
![[backprop.png]]

struct MLP {

int n_layers;
int* layers; //array of hidden layer sizes
float** weights; //array of pointers to arrays of floats (linearized matrices)
float** biases; //array of pointers to arrays of floats
float** activations; //array of pointers to arrays of floats
float** logits; //array of pointers to arrays of floats
float** gradients; //array of pointers to arrays of floats (linearizzed matrices)
float** deltas; //array of pointers to arrays of floats
float* inputs; //array of floats
float* labels; //array of floats

};