#ifndef MLP_H
#define MLP_H

/*structure of MLP
- int num_layers; // numebr of hidden layers in neural network
- int *layers_sizes; // array where each entry is the # of nodes in that layer
    ex: layer1 = 3 nodes, layer2 = 6, layer3 = 10 then hidden_layer_size=[3,6,10] and num_hidden_layers=3
- neuron_activation[layer][node] = array of pointers(without comprehending input layer) to doubles
- double **weights[layer][hideen_layers_size[layer]*hideen_layers_size[layer-1]] = (array of pointers (layer) (does not comprehend input llayer) to linearized 2D matrix)
               = weight between a node of the current layer and a node of previous layer
- double **biases array of pointers to array of biases [layer][node] does not comprehend input llayer)*/
typedef struct MLP {
    int num_layers; // numebr of hidden layers in neural network
    int *layers_sizes; // array where each entry is the # of nodes in that layer
    double **neuron_activations;//neuron activations of each layer

    // weight between a node of the current layer and a node of previous layer, (it start from first hidden layer)
    // [layer][hideen_layers_size[layer]*hideen_layers_size[layer-1]]
    double **weights; //note input lauer doesnt have weights
    double **biases;// note: input layer doesn0t have bias
} MLP;

MLP *createMLP(int num_layers, int *layers_size);
void initializeXavier(double *weights, int in, int out);
void printMLP(const MLP *mlp);
#endif // MLP_H