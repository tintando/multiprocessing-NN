#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib> // Include the <cstdlib> header to define the malloc function
#include <math.h>

typedef struct{
    double* data;
}Sylly;

typedef struct{
    Sylly *sylly;
}Sylly2;

Sylly* createSilly(){
    Sylly* silly = (Sylly*)malloc(sizeof(Sylly));
    silly->data = (double*)malloc(10*sizeof(double));
    for (int i = 0; i < 10; i++){
        silly->data[i] = i;
    }
    return silly;
}

void createSyllis2(Sylly2* sylly2, Sylly* sylly, int num_syllis){
    
    for (int i = 0; i < num_syllis; i++){
        sylly2[i].sylly = sylly;
    }

}

int main(){
    Sylly* sylly = createSilly();
    Sylly2* sylly2 = (Sylly2*)malloc(20*sizeof(Sylly2));//array of sylly2
    createSyllis2(sylly2, sylly, 20);
    double a = sylly2[0].sylly->data[0];
    printf("a = %lf\n", a);
}