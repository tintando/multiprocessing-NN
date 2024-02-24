#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib> // Include the <cstdlib> header to define the malloc function
#include<math.h>

// To compile this file, run the following command:
//g++ Accomulators_optimization.cpp ../src/mlp.cpp -o accomulators -lpthread

/*
- double **weights[layer][hideen_layers_size[layer]*hideen_layers_size[layer-1]] = (array of pointers (layer) (does not comprehend input llayer) to linearized 2D matrix)
               = weight between a node of the current layer and a node of previous layer
*/

#define NUM_THREADS 3
#define NUM_ACC_THREADS 10000
#define TOTAL_ITERATIONS_SUM 1000

pthread_cond_t cond_waitfor_main_accomulatorWB = PTHREAD_COND_INITIALIZER;
pthread_cond_t cond_waitfor_accomulatorWB_main = PTHREAD_COND_INITIALIZER;
pthread_cond_t cond_waitfor_main_accomulatorWBpause = PTHREAD_COND_INITIALIZER;
pthread_mutex_t lock_accomulatorWB = PTHREAD_MUTEX_INITIALIZER;
int counter_accomulatorWB_finished = 0;
int flag_mainworking_accoumulatorWB = 0;
int flag_start_accomulatorsWB = 0;
int flag_stop_accomulatorsWB = 0;

typedef struct {
    long thread_id;
    double**** weights_threads; //pointer to an array of pointers(thread) to an array of pointers(layer) to an array of pointers to weights
    double*** weights_global; //pointer to list of weights of main thread (pointer to an array of pointers(layer) to an array of weights)
    int start_layer_weight;
    int start_weight;
    int weight_counter_max;
    int num_layers; 
    int** layers_size; //pointer to array of pointers(layer) to layers sizes
    int num_working_threads; //how many working threads there are (if there are more threads than wights in the NN)
} Thread_args_accomulator;

//popular way to initialize weights
//helps in keeping the signal from the input to flow well into the deep network.
void initializeXavier(double *weights, int in, int out) {
    // in = prev_layer_size
    // out = layer size
    // weights = weights between nodes of previous and current layer
    double limit = sqrt(6.0 / (in + out));
    for (int i = 0; i < in * out; i++) {
        weights[i] = (rand() / (double)(RAND_MAX)) * 2 * limit - limit;
    }
}

double*** createWeights(int num_layers, int* layers_size) {
    double*** weights = (double***) malloc(NUM_THREADS * sizeof(double**));
    for (int t = 0; t < NUM_THREADS; t++) {
        weights[t] = (double**) malloc(num_layers * sizeof(double*));
        for (int i = 1; i<num_layers; i++) {
            weights[t][i] = (double*) malloc(layers_size[i] * layers_size[i-1] * sizeof(double));
            initializeXavier(weights[t][i], layers_size[i-1], layers_size[i]);
        }
        // Print weights
        for (int i = 1; i < num_layers; i++) { // Start from 1 since weights are between layers
            printf("Weights to Layer %d: \n", i);
            for (int j = 0; j < layers_size[i]; j++) {
                for (int k = 0; k < layers_size[i-1]; k++) {
                    printf("W[%d][%d][%d]: %lf ",t, j, k, weights[t][i][j * layers_size[i-1] + k]);
                }
                printf("\n");
            }       
        }
    }
    return weights;
}
 void *thread_function(void* voidArgs){
    Thread_args_accomulator *args = (Thread_args_accomulator*)voidArgs;
    long thread_id = args->thread_id;
    printf("[%d] starts from layer %d\n", thread_id, args->start_layer_weight);
    if (args->start_layer_weight == -1) return NULL;

    int counter;
    int next_layer_flag;

    // pthread_mutex_lock(&lock_accomulatorWB);
    //     while(flag_start_accomulatorsWB == 0){
    //         printf("Thread #%ld waiting for main to start\n", thread_id);
    //         pthread_cond_wait(&cond_waitfor_main_accomulatorWB, &lock_accomulatorWB);
    //     }
    // pthread_mutex_unlock(&lock_accomulatorWB);

    while(1){

        //wait until main thread says to start
        pthread_mutex_lock(&lock_accomulatorWB);
        while (flag_start_accomulatorsWB == 0){
            printf("Thread #%ld waiting on cond_waitfor_main_accomulatorWB because flag_start_accomulatorsWB = 0\n", thread_id);
            pthread_cond_wait(&cond_waitfor_main_accomulatorWB, &lock_accomulatorWB);
        }
        pthread_mutex_unlock(&lock_accomulatorWB);
        if (flag_stop_accomulatorsWB == 1) break;

        counter = 0;
        next_layer_flag=0;

        //thread work
        for (int i = args->start_layer_weight; i <= args->num_layers; i++) {
            //printf("thread[%d] is working on layer[%d]\n",thread_id, i);

            if (i==-1) {printf("thread[%d] is done because it starts weights from -1\n", thread_id); break;}//if it is done, it has to break the loop

            //if the thread changes layer, it has to start from the first weight, otherwise it has to start from its weight range
            for (int w = (next_layer_flag)? 0 : args->start_weight; w < (*args->layers_size)[i] * (*args->layers_size)[i-1]; w++) {
                printf("thread[%d] is working on weight[%d][%d]\n",thread_id, i, w);
                
                //if summed all the weights of the thread, it is done
                if (counter == args->weight_counter_max){
                    //printf("thread[%d] was joking,\n", thread_id);
                    break;
                }

                //sum the weights
                for (int t = 0; t < NUM_THREADS; t++){
                    (*args->weights_global)[i][w] += (*args->weights_threads)[t][i][w];
                    //printf("thread [%d] is summing weight[%d][%d][%d] = %lf\n",thread_id, t, i, w, (*args->weights_threads)[t][i][w]);
                }
                counter++;
                //printf("counter = %d\n", counter);
            }

            //if it is done, it has to break the loop
            if (counter == args->weight_counter_max){
                printf("thread[%d] is done\n", thread_id);
                    break;
                }
            //if it is not done, it has to start from the first weight of the next layer
            else next_layer_flag = 1;
        }
        
        pthread_mutex_lock(&lock_accomulatorWB);
        counter_accomulatorWB_finished++;//incrase the counter and in case of the last thread, signal the main
        if (counter_accomulatorWB_finished == args->num_working_threads) {pthread_cond_signal(&cond_waitfor_accomulatorWB_main); printf("Thread #%ld signals main\n", thread_id);}
        printf("thread #%d waits on cond_waitfor_main_accomulatorWBpause because increases counter to %d\n", thread_id, counter_accomulatorWB_finished);
        pthread_cond_wait(&cond_waitfor_main_accomulatorWBpause, &lock_accomulatorWB);
        pthread_mutex_unlock(&lock_accomulatorWB);
        //


    }
    if (args->start_layer_weight == -1) printf("thread[%d] is done because layer is -1\n", thread_id);
    return NULL;
 }


 void accessWeights(double**** weight){
    /*
    weight is a double****, a pointer to a double***.
    *weight dereferences it to a double***, which is the first level of dereferencing, leading to a pointer to a pointer to a pointer to double.
    (*weight)[0] accesses the first double** in the array of double**.
    (*weight)[0][0] accesses the first double* in the array of double*.
    Finally, (*weight)[0][0][0] correctly accesses the first double value.
    */
    double w = (*weight)[0][1][0]; // Correctly dereferences to the first element
    printf("weight[0][0][0] = %lf\n", w);
 }

 void accessargWeights(void* voidArgs){
    Thread_args_accomulator *args = (Thread_args_accomulator*)voidArgs;

    double my_weight = (*args->weights_threads)[0][1][0];
 }

int main() {

    int layers_size_init[] = {4, 3, 2, 1};
    int num_layers = 4;
    int* layers_size = (int*)malloc(num_layers * sizeof(int));
    for (int i = 0; i < num_layers; i++) {//populating the aray of layers sizes
        layers_size[i] = layers_size_init[i];
    }

    double*** weights = createWeights(num_layers, layers_size);


    //-------------finished generating weights for each thread----------------
    // now we have to sum the weights of each thread
    // split evenly the work of summing the weights of each thread to the threads

    int total_weights = 0;
    for(int i = 1; i < num_layers; i++){
        total_weights += layers_size[i] * layers_size[i-1];
        printf("layer[%d] has %d weights connecting to previous layer\n", i, layers_size[i] * layers_size[i-1]);
    }
    printf("total_weights = %d\n", total_weights);

    int weights_per_thread = total_weights/NUM_ACC_THREADS;
    printf("weights_per_thread = %d\n", weights_per_thread);
    int remainder = total_weights%NUM_ACC_THREADS;
    printf("remainder = %d\n", remainder);

    Thread_args_accomulator* thread_args_accomulatorWB = (Thread_args_accomulator*)malloc(NUM_ACC_THREADS* sizeof(Thread_args_accomulator));

    int start_flag = 1;
    int thread_id = 0;
    int count = 0;
    for (int i = 1; i<num_layers; i++){
        for (int w=0; w<layers_size[i]*layers_size[i-1]; w++){//iterate trough the weights
            if (start_flag) {
                printf("thread %d starts from [%d][%d]\n", thread_id, i, w); 
                thread_args_accomulatorWB[thread_id].start_layer_weight = i;
                thread_args_accomulatorWB[thread_id].start_weight = w;
                start_flag = 0;
            }
            count++;
            if (count >= ((remainder>0) ? weights_per_thread + 1 : weights_per_thread)){
                printf("thread %d finishs at [%d][%d]\n", thread_id, i, w);
                thread_args_accomulatorWB[thread_id].weight_counter_max = count;
                start_flag = 1;
                remainder --;
                thread_id ++;
                count = 0;
            }
        }
    }
    int num_working_weights_threads = thread_id;
    for (int i = 0; i < NUM_ACC_THREADS; i++){
        if (i >= num_working_weights_threads) thread_args_accomulatorWB[i].start_layer_weight = -1;
        thread_args_accomulatorWB[i].num_working_threads = num_working_weights_threads;
    }

    //------------each thread now knows the range of weights it has to sum----------------
    // the same could be done with biases, but since this is only an example, we will only sum the weights

    //print weights using the index of each thread
    int next_layer_flag = 0;

    double** global_weights2 = (double**) malloc(NUM_THREADS * sizeof(double*));
    for (int i = 1; i<num_layers; i++){
        global_weights2[i] = (double*)calloc(layers_size[i] * layers_size[i-1], sizeof(double));
    }
    
    for (int thread_id=0; thread_id<NUM_ACC_THREADS; thread_id++){//thread_id simulates multithreading
        printf("thread[%d] will sum weights from [%d][%d] \n", thread_id, thread_args_accomulatorWB[thread_id].start_layer_weight, thread_args_accomulatorWB[thread_id].start_weight);
        
        int counter = 0;
        next_layer_flag = 0;

        //start summing the weights from the range of each thread
        for (int i = thread_args_accomulatorWB[thread_id].start_layer_weight; i <= num_layers; i++) {
            printf("thread[%d] is working on layer[%d]\n",thread_id, i);

            //if the thread changes layer, it has to start from the first weight, otherwise it has to start from its weight range
            for (int w = (next_layer_flag)? 0 : thread_args_accomulatorWB[thread_id].start_weight; w < layers_size[i]*layers_size[i-1]; w++) {
                printf("thread[%d] is working on weight[%d][%d]\n",thread_id, i, w);
                
                //if summed all the weights of the thread, it is done
                if (counter == thread_args_accomulatorWB[thread_id].weight_counter_max){
                    printf("thread[%d] was joking,\n", thread_id);
                    break;
                }

                //sum the weights
                for (int k = 0; k < TOTAL_ITERATIONS_SUM; k++){
                for (int t = 0; t < NUM_THREADS; t++){
                    double weight = (weights[t][i][w]);
                    printf("thread [%d] is summing weight[%d][%d][%d] = %lf\n",thread_id, t, i, w, weight);
                    global_weights2[i][w] += weight;
                }
                }
                counter++;
                printf("counter = %d\n", counter);
            }

            //if it is done, it has to break the loop
            if (counter == thread_args_accomulatorWB[thread_id].weight_counter_max){
                printf("thread[%d] is done\n", thread_id);
                    break;
                }
            //if it is not done, it has to start from the first weight of the next layer
            else next_layer_flag = 1;
        }
    }
    thread_args_accomulatorWB[0].weights_threads = &weights;
        accessWeights(&weights);
    printf("STARTING MULTITHREADED-----------------------------------------------------------------------------");
    double** global_weights = (double**) malloc(NUM_THREADS * sizeof(double*));
    for (int i = 1; i<num_layers; i++){
        global_weights[i] = (double*)calloc(layers_size[i]*layers_size[i-1], sizeof(double));
    }

    pthread_t threads[NUM_ACC_THREADS];
    for (long t = 0; t < NUM_ACC_THREADS; t++) {
        printf("Creating thread %d\n", t);
        thread_args_accomulatorWB[t].thread_id = t;
        thread_args_accomulatorWB[t].weights_threads = &weights;
        thread_args_accomulatorWB[t].weights_global = &global_weights;
        thread_args_accomulatorWB[t].num_layers = num_layers;
        thread_args_accomulatorWB[t].layers_size = &layers_size;
        pthread_create(&threads[t], NULL, thread_function, (void*)&(thread_args_accomulatorWB[t]));
    }
    int i = 0;
    flag_mainworking_accoumulatorWB = 1;
    printf("num_working_weights_threads = %d\n", num_working_weights_threads);

    while(1){
 //-------------------work
        printf("-\n-\n-\n-\n");
        printf("Main working\n");
        printf("-\n-\n-\n-\n"); 
        sleep(0);
        //------------------------------------------------
        pthread_mutex_lock(&lock_accomulatorWB);
        //start threads
        printf("main set flag_start_accomulatorsWB to 1\n");
        flag_start_accomulatorsWB = 1;
        printf("main sets flag_mainworking_accoumulatorWB to 0\n");
        printf("main signal threads waiting on cond_waitfor_main_accomulatorWB\n");
        flag_mainworking_accoumulatorWB = 0;
        pthread_cond_broadcast(&cond_waitfor_main_accomulatorWB);
        pthread_mutex_unlock(&lock_accomulatorWB);

        //------pause main thread until all threads finish
        pthread_mutex_lock(&lock_accomulatorWB);
        while (counter_accomulatorWB_finished < num_working_weights_threads){
            printf("Main waiting on cond_waitfor_accomulatorWB_main because counter_accomulatorWB_finished = %d\n", counter_accomulatorWB_finished);
            pthread_cond_wait(&cond_waitfor_accomulatorWB_main, &lock_accomulatorWB);
        }
        pthread_mutex_unlock(&lock_accomulatorWB);
        pthread_mutex_lock(&lock_accomulatorWB);
        printf("all threads finished and waiting on cond_waitfor_main_accomulatorWBpause\n");
        printf("Main resumes because all threads finished\n");
        printf("main sets flag_mainworking_accoumulatorWB to 1\n");
        printf("main sets flag_start_accomulatorsWB to 0\n");
        printf("main sets counter_accomulatorWB_finished to 0\n");
        counter_accomulatorWB_finished = 0;//resetting the counter of finished accomulators
        flag_start_accomulatorsWB = 0; // pause accomulatorsWB 
        flag_mainworking_accoumulatorWB = 1;
        printf("main thread signals threads waiting on cond_waitfor_main_accomulatorWBpause\n");
        pthread_cond_broadcast(&cond_waitfor_main_accomulatorWBpause);
        pthread_mutex_unlock(&lock_accomulatorWB);
        
        
        i++;
        if (i == TOTAL_ITERATIONS_SUM) {
            printf("Main finished\n"); 
            pthread_mutex_lock(&lock_accomulatorWB);
            flag_stop_accomulatorsWB = 1;
            flag_start_accomulatorsWB = 1; //to make htem go out of the loop
            pthread_cond_broadcast(&cond_waitfor_main_accomulatorWB);
            pthread_mutex_unlock(&lock_accomulatorWB);
            break;}
    }
    printf("joining threads\n");
    for (i=0; i<NUM_ACC_THREADS; i++){
    pthread_join(threads[i], NULL);
    }
    printf("threads joined\n");
    
    int c = 1;
    // Print weights
    printf("GLOBAL MULTITHREAD");
        for (int i = 1; i < num_layers; i++) { // Start from 1 since weights are between layers
            printf("Weights to Layer %d: \n", i);
            for (int j = 0; j < layers_size[i]; j++) {
                for (int k = 0; k < layers_size[i-1]; k++) {
                    printf("W[%d][%d]: %lf ", j, k, global_weights[i][j * layers_size[i-1] + k]);
                }
                printf("\n");
            }       
        }

    printf("GLOBAL SERIAL");
        for (int i = 1; i < num_layers; i++) { // Start from 1 since weights are between layers
            printf("Weights to Layer %d: \n", i);
            for (int j = 0; j < layers_size[i]; j++) {
                for (int k = 0; k < layers_size[i-1]; k++) {
                    printf("W[%d][%d]: %lf ", j, k, global_weights2[i][j * layers_size[i-1] + k]);
                }
                printf("\n");
            }       
        }
    printf("\n%d\n", c);

}
    