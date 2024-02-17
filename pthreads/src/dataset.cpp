#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string.h>

#define N_FEATURES 8
#define N_LABELS 1

//contains features and label for a sample
struct Sample {
    float features[N_FEATURES];
    float label;
};

typedef struct Data {
    double** samples; 
    double** targets;
    int size; // number of samples
}Data;

typedef struct Dataset {
    Data train;
    Data test;
    Data validation;
}Dataset;

Sample* readDataset(const char* filePath, int* n_samples) {
    // Open the file
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filePath << std::endl;
        return nullptr;
    }

    // Determine the number of samples
    int numSamples = 0;
    std::string line;
    while (std::getline(file, line)) {
        numSamples++;
    }
    file.clear();
    file.seekg(0, std::ios::beg);
    *n_samples = numSamples;
    // Allocate memory for the array of structs
    Sample* samples = new Sample[numSamples];

    // Read the data from the file
    for (int i = 0; i < numSamples; i++) {
        std::getline(file, line);
        std::istringstream iss(line);
        
        // Read the features
        for (int j = 0; j < N_FEATURES; j++) {
            std::string feature;
            std::getline(iss, feature, ',');
            samples[i].features[j] = std::stof(feature);
        }

        // Read the label
        std::string label;
        std::getline(iss, label, ',');
        samples[i].label = std::stof(label);
    }

    // Close the file
    file.close();

    return samples;
}

void freeDataset(Dataset* dataset) {
    // Free train samples and targets
    for (int i = 0; i < dataset->train.size; i++) {
        free(dataset->train.samples[i]);
        free(dataset->train.targets[i]);
    }
    free(dataset->train.samples);
    free(dataset->train.targets);

    // Free test samples and targets
    for (int i = 0; i < dataset->test.size; i++) {
        free(dataset->test.samples[i]);
        free(dataset->test.targets[i]);
    }
    free(dataset->test.samples);
    free(dataset->test.targets);

    // If you have validation data, free it similarly here
}

void printSamples(Sample* samples, int n) {
    printf("Size of Sample: %lu bytes\n", sizeof(Sample));
    for (int i = 0; i < n; i++) {
        printf("Sample %d: ", i);
        for (int j = 0; j < N_FEATURES; j++) {
            printf("%.2f ", samples[i].features[j]);
        }
        printf("Label: %.2f\n", samples[i].label);
    }
}

void printData(Data data) {
    printf("Samples:\n");
    for (int i = 0; i < data.size; i++) {
        printf("Sample %d: ", i + 1);
        for (int j = 0; j < N_FEATURES; j++) {
            printf("%f ", data.samples[i][j]);
        }
        printf("\n");
    }

    printf("Targets:\n");
    for (int i = 0; i < data.size; i++) {
        printf("Target %d: ", i + 1);
        for (int j = 0; j < N_LABELS; j++) {
            printf("%f ", data.targets[i][j]);
        }
        printf("\n");
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
        delete[] samples; // Assuming samples need to be freed here
        return;
    }

    // Copy data from samples to h_features and h_labels
    for (int i = 0; i < *n_samples; i++) {
        memcpy(h_features + i * N_FEATURES, samples[i].features, N_FEATURES * sizeof(float));
        h_labels[i] = samples[i].label;
    }

    // Assuming the responsibility to free samples is here
    // Remember to free the samples' features if they're dynamically allocated
    delete[] samples;

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


// Splits the dataset into train, validation, and test sets
Dataset splitDataset(int n_samples, double*** dataset, double*** targets){

    int train_size = (int)(n_samples*80/100), test_size = n_samples - train_size;
    double **train_data = (double**) malloc(train_size * sizeof(double*));
    double **train_targets = (double**) malloc(train_size * sizeof(double*));
    double **test_data = (double**) malloc(test_size * sizeof(double*));
    double **test_targets = (double**) malloc(test_size * sizeof(double*));

    for (int i = 0; i < train_size; i++) {
        train_data[i] = (double*) malloc(N_FEATURES * sizeof(double));
        train_targets[i] = (double*) malloc(N_LABELS * sizeof(double));
        for (int j = 0; j < N_FEATURES; j++) {
            train_data[i][j] = (*dataset)[i][j];
        }
        train_targets[i][0] = (*targets)[i][0];
    }

    for (int i = 0; i < test_size; i++) {
        test_data[i] = (double*) malloc(N_FEATURES * sizeof(double));
        test_targets[i] = (double*) malloc(N_LABELS * sizeof(double));
        for (int j = 0; j < N_FEATURES; j++) {
            test_data[i][j] = (*dataset)[i + train_size][j];
        }
        test_targets[i][0] = (*targets)[i + train_size][0];
    }

    Dataset result;
    result.train.samples = train_data;
    result.train.targets = train_targets;
    result.train.size = train_size;
    result.test.samples = test_data;
    result.test.targets = test_targets;
    result.test.size = test_size;
    // Note: If you add validation data, initialize result.validation here

    return result;
}