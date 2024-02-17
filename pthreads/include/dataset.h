#ifndef DATASET_H
#define DATASET_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string.h>


struct Sample {
    float features[8];
    float label;
};

    // double** samples; //array of pointers (sample #) to array of feautres
    // double** targets; //array of pointers (sample #) to array of targets
    // int size; // Number of samples
typedef struct Data {
    double** samples; //array of pointers (sample #) to array of feautres
    double** targets; //array of pointers (sample #) to array of targets
    int size; // Number of samples
} Data;
 
    // Data train;
    // Data test;
    // Data validation;
typedef struct Dataset {
    Data train;
    Data test;
    Data validation;
};

// Data handling functions
void printData(Data data);
void loadAndPrepareDataset(const char* filename, double ***dataset, double ***targets, int *n_samples);
void shuffleDataset(double ***dataset, double ***targets, int n_samples);
void freeDataset(Dataset* dataset);
Dataset splitDataset(int n_samples, double*** dataset, double*** targets);
void freeData(Data* data);//to implement

// Data loading functions
Sample* readDataset(const char* filePath, int* n_samples);
void printSamples(Sample* samples, int n);
#endif // DATA_LOADING_H