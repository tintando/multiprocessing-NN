#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>

void myFunction(double** arr, int rows, int cols) {
    // Example usage of arr
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Access the element at [i][j] and do something with it, e.g., print it
            printf("%f ", arr[i][j]);
        }
        printf("\n");
    }
}


int main() {
    int rows = 3; // Number of rows
    int cols = 2; // Number of columns

    // Dynamically allocate an array of pointers to double
    double** arr = (double**) malloc(rows * sizeof(double*));

    // Dynamically allocate each row
    for (int i = 0; i < rows; ++i) {
        arr[i] = (double*)malloc(cols * sizeof(double));
    }

    // Example initialization of arr
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            arr[i][j] = i + j; // Just an example initialization
        }
    }

    // Now, pass arr to the function
    myFunction(arr, rows, cols);

    // Don't forget to deallocate memory to prevent memory leaks
    for (int i = 0; i < rows; ++i) {
        free(arr[i]); // Deallocate each row
    }
    free(arr); // Deallocate the array of pointers

    return 0;
}
