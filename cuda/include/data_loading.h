#ifndef DATA_LOADING_H
#define DATA_LOADING_H

struct Sample {
    float features[8];
    float label;
};

Sample* readDataset(const char* filePath);
void printSamples(Sample* samples, int n);
#endif // DATA_LOADING_H