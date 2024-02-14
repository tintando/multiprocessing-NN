#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#define N_FEATURES 8
#define N_LABELS 1

//contains features and label for a sample
struct Sample {
    float features[N_FEATURES];
    float label;
};

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