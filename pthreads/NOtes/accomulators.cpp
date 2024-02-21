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

#define NUM_THREADS 5
#define NUM_ACC_THREADS 6

pthread_cond_t cond_waitfor_accumulatorWB_main = PTHREAD_COND_INITIALIZER; //condition vairable to signal the main threads all the accomulators for weight and bias are done
pthread_cond_t cond_waitfor_main_accomulatorWB = PTHREAD_COND_INITIALIZER; //condition variable to signal the threads that main thread is ready to make them continue
pthread_cond_t cond_start_accomulatorsWB = PTHREAD_COND_INITIALIZER; // Condition variable to signal start of work
pthread_mutex_t lock_counter_accomulatorsWB_done = PTHREAD_MUTEX_INITIALIZER;
int counter_accomulatorsWB_done = 0;
int flag_start_accomulatorsWB = 0; // Flag to indicate workers can start
int flag_stop_accomulatorsWB = 0; // Flag to indicate workers should stop


typedef struct {
    long thread_id;
    double**** weights_threads; //pointer to an array of pointers(thread) to an array of pointers(layer) to an array of pointers to weights
    double*** weights_global; //pointer to list of weights of main thread (pointer to an array of pointers(layer) to an array of weights)
    int start_layer_weight;
    int start_weight;
    int weight_counter_local_max;
    int num_layers; 
    int** layers_size; //pointer to array of pointers(layer) to layers sizes
} Thread_args_sum;



//popular way to initialize weights
//helps in keeping the signal from the input to flow well into the deep network.
void initializeXavier(double *weights, int in, int out) {
    double limit = sqrt(6.0 / (in + out));
    for (int i = 0; i < in * out; i++) {
        weights[i] = (rand() / (double)(RAND_MAX)) * 2 * limit - limit;
    }
}



//returns a pointer to an array of pointers (thread) to an array of pointers(layer) to an array of weights
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
            //printf("Weights to Layer %d: \n", i);
            for (int j = 0; j < layers_size[i]; j++) {
                for (int k = 0; k < layers_size[i-1]; k++) {
                    //printf("W[%d][%d][%d]: %lf ",t, j, k, weights[t][i][j * layers_size[i-1] + k]);
                }
                //printf("\n");
            }       
        }
    }
    return weights;
}




 void *thread_function(void* voidArgs){
    Thread_args_sum *args = (Thread_args_sum*)voidArgs;
    long thread_id = args->thread_id;
    printf("[%d] start from weight[%d][%d] and sum %d weights\n", thread_id, args->start_layer_weight, args->start_weight, args->weight_counter_local_max);

    
    //wait until main thread says to start
    pthread_mutex_lock(&lock_counter_accomulatorsWB_done);
    while (!flag_start_accomulatorsWB) { // Wait until start_flag is set
        printf("[%d] sleeping on start_accomulatorsWB_cond\n", args->thread_id);
        pthread_cond_wait(&cond_start_accomulatorsWB, &lock_counter_accomulatorsWB_done);
        printf("[%d] wake up\n", thread_id);
    }
    pthread_mutex_unlock(&lock_counter_accomulatorsWB_done);

    int counter_local;//keeps track of how many weights have been summed
    int next_layer_flag;//tells if the thread has to start from the first weight of the next layer

    while(1){
        //checking if the stop flag is true, in case break the loop
        pthread_mutex_lock(&lock_counter_accomulatorsWB_done);
            printf("[%ld] is checking flag_stop_accomulatorsWB\n", args->thread_id);
            if(flag_stop_accomulatorsWB) break;
        pthread_mutex_unlock(&lock_counter_accomulatorsWB_done);

        //waiting for signal to continue from main thread
        pthread_mutex_lock(&lock_counter_accomulatorsWB_done);
            printf("[%ld] is waiting on cond_waitfor_main_accomulatorWB\n", args->thread_id);
            pthread_cond_wait(&cond_waitfor_main_accomulatorWB, &lock_counter_accomulatorsWB_done);
        pthread_mutex_unlock(&lock_counter_accomulatorsWB_done);

        counter_local = 0; 
        next_layer_flag = 0;

        //start summing the weights from the range of each thread
        for (int i = args->start_layer_weight; i < args->num_layers; i++) {
            //printf("thread[%d] is working on layer[%d]\n",thread_id, i);

            //if the thread changes layer, it has to start from the first weight, otherwise it has to start from its weight range
            for (int w = (next_layer_flag)? 0 : args->start_weight; w < (*args->layers_size)[i] * (*args->layers_size)[i-1]; w++) {
    
                printf("[%d] working on weight[%d][%d]\n",thread_id, i, w);
                //sum the weights
                for (int t = 0; t < NUM_THREADS; t++){
                    (*args->weights_global)[i][w] += (*args->weights_threads)[t][i][w];
                    //printf("thread [%d] is summing weight[%d][%d][%d] = %lf\n",thread_id, t, i, w, (*args->weights_threads)[t][i][w]);
                }
                counter_local++;
                //printf("[%d] finished working on weight[%d][%d], increasing counter_local:%d", thread_id, i, w, counter_local);
                //if summed all the weights of the thread, it is done
                if (counter_local == args->weight_counter_local_max){
                    printf("[%d] counter_local is %d , i stop at weight[%d][%d]\n", thread_id, counter_local, i, w);
                    break;
                }
            }

            //if it is done, it has to stop summing
            if (counter_local == args->weight_counter_local_max){
                printf("[%d] i'm done\n", thread_id);
                    break;
                }
            //if it is not done, it has to start from the first weight of the next layer
            else next_layer_flag = 1;
            printf("[%d] will continue from weight[%d][%d]\n", thread_id, i+1, 0);
        }
        

        // Signal main thread that this worker is done
        pthread_mutex_lock(&lock_counter_accomulatorsWB_done);
            printf("[%d] finished, incrementing counter_accomulatorsWB_done to %d\n", args->thread_id, counter_accomulatorsWB_done+1);
            counter_accomulatorsWB_done++;
            //if it is the last thread, signal the main thread
            if (counter_accomulatorsWB_done == NUM_ACC_THREADS-1){
                printf("Thread %ld is signaling cond_waitfor_accumulatorWB_main\n", args->thread_id);
                pthread_cond_signal(&cond_waitfor_accumulatorWB_main);
            }
        pthread_mutex_unlock(&lock_counter_accomulatorsWB_done);

    }

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
    //printf("weight[0][0][0] = %lf\n", w);
 }

 void accessargWeights(void* voidArgs){
    Thread_args_sum *args = (Thread_args_sum*)voidArgs;

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
        //printf("layer[%d] has %d weights connecting to previous layer\n", i, layers_size[i] * layers_size[i-1]);
    }

    int weights_per_thread = total_weights/NUM_THREADS;
    //printf("weights_per_thread = %d\n", weights_per_thread);
    int remainder = total_weights%NUM_THREADS;
    //printf("remainder = %d\n", remainder);

    Thread_args_sum* thread_args_sum = (Thread_args_sum*)malloc(NUM_THREADS* sizeof(Thread_args_sum));

    int start_flag = 1;
    int thread_id = 0;
    int count = 0;
    for (int i = 1; i<num_layers; i++){
        for (int w=0; w<layers_size[i]*layers_size[i-1]; w++){//iterate trough the weights
            if (start_flag) {
                //printf("thread %d starts from [%d][%d]\n", thread_id, i, w); 
                thread_args_sum[thread_id].start_layer_weight = i;
                thread_args_sum[thread_id].start_weight = w;
                start_flag = 0;
            }
            count++;
            if (count >= ((remainder>0) ? weights_per_thread + 1 : weights_per_thread)){
                //printf("thread %d finishs at [%d][%d]\n", thread_id, i, w);
                thread_args_sum[thread_id].weight_counter_local_max = count;
                start_flag = 1;
                remainder --;
                thread_id ++;
                count = 0;
            }
        }
    }

    //------------each thread now knows the range of weights it has to sum----------------
    // the same could be done with biases, but since this is only an example, we will only sum the weights

    //print weights using the index of each thread
    int next_layer_flag = 0;

    double** global_weights2 = (double**) malloc(NUM_THREADS * sizeof(double*));
    for (int i = 1; i<num_layers; i++){
        global_weights2[i] = (double*)calloc(layers_size[i] * layers_size[i-1], sizeof(double));
    }
    
    for (int thread_id=0; thread_id<NUM_THREADS; thread_id++){//thread_id simulates multithreading
        //printf("thread[%d] will sum weights from [%d][%d] \n", thread_id, thread_args_sum[thread_id].start_layer_weight, thread_args_sum[thread_id].start_weight);
        
        int counter_local = 0;
        next_layer_flag = 0;

        //start summing the weights from the range of each thread
        for (int i = thread_args_sum[thread_id].start_layer_weight; i <= num_layers; i++) {
            //printf("thread[%d] is working on layer[%d]\n",thread_id, i);

            //if the thread changes layer, it has to start from the first weight, otherwise it has to start from its weight range
            for (int w = (next_layer_flag)? 0 : thread_args_sum[thread_id].start_weight; w < layers_size[i]*layers_size[i-1]; w++) {
                //printf("thread[%d] is working on weight[%d][%d]\n",thread_id, i, w);
                
                //if summed all the weights of the thread, it is done
                if (counter_local == thread_args_sum[thread_id].weight_counter_local_max){
                    //printf("thread[%d] was joking,\n", thread_id);
                    break;
                }

                //sum the weights
                for (int t = 0; t < NUM_THREADS; t++){
                    double weight = (weights[t][i][w]);
                    //printf("thread [%d] is summing weight[%d][%d][%d] = %lf\n",thread_id, t, i, w, weight);
                    global_weights2[i][w] += weight * 50;
                }
                counter_local++;
                //printf("counter_local = %d\n", counter_local);
            }

            //if it is done, it has to break the loop
            if (counter_local == thread_args_sum[thread_id].weight_counter_local_max){
                //printf("thread[%d] is done\n", thread_id);
                    break;
                }
            //if it is not done, it has to start from the first weight of the next layer
            else next_layer_flag = 1;
        }
    }
    thread_args_sum[0].weights_threads = &weights;
        accessWeights(&weights);
    //printf("STARTING MULTITHREADED-----------------------------------------------------------------------------");
    double** global_weights = (double**) malloc(NUM_THREADS * sizeof(double*));
    for (int i = 1; i<num_layers; i++){
        global_weights[i] = (double*)calloc(layers_size[i]*layers_size[i-1], sizeof(double));
    }

    pthread_t threads[NUM_THREADS];
    for (long t = 0; t < NUM_THREADS; t++) {
        //printf("Creating thread %d\n", t);
        thread_args_sum[t].thread_id = t;
        thread_args_sum[t].weights_threads = &weights;
        thread_args_sum[t].weights_global = &global_weights;
        accessWeights(thread_args_sum[t].weights_threads);
        thread_args_sum[t].num_layers = num_layers;
        thread_args_sum[t].layers_size = &layers_size;
        pthread_create(&threads[t], NULL, thread_function, (void*)&(thread_args_sum[t]));
    }
//start main thread work!!!!!!!!

    // Signal worker threads to start work
    printf("Main thread is starting all WB threads waiting on lock_counter_accomulatorsWB_done by setting flag to 1\n");
    pthread_mutex_lock(&lock_counter_accomulatorsWB_done);
    flag_start_accomulatorsWB = 1;
    pthread_cond_broadcast(&cond_start_accomulatorsWB);
    pthread_mutex_unlock(&lock_counter_accomulatorsWB_done);

    for (int i = 0; i<50; i++){
        printf("i=[%d]\n",i);
        // Main thread work before signaling workers to start
        
        // Signal worker threads to continue
        //printf("Main thread is ready to continue, broadcasting waiting working threads\n");
        pthread_mutex_lock(&lock_counter_accomulatorsWB_done);
            if (i == 50){
                flag_stop_accomulatorsWB = 1;
            }
        pthread_mutex_unlock(&lock_counter_accomulatorsWB_done);
        
        printf((i==50)?  "Main thread is done since i = %d\n" : "Main thread is continuing since i = %d\n", i);

        pthread_mutex_lock(&lock_counter_accomulatorsWB_done);
            printf("main thread is broadcasting cond_waitfor_main_accomulatorWB");
            pthread_cond_broadcast(&cond_waitfor_main_accomulatorWB); // Use broadcast to signal all waiting threads
        pthread_mutex_unlock(&lock_counter_accomulatorsWB_done);

        // Wait for all worker threads to signal they are done
        pthread_mutex_lock(&lock_counter_accomulatorsWB_done);
        while (counter_accomulatorsWB_done < NUM_ACC_THREADS-1){
            printf("Main thread is waiting on cond_waitfor_accumulatorWB_main\n");
            pthread_cond_wait(&cond_waitfor_accumulatorWB_main, &lock_counter_accomulatorsWB_done);
        }
        pthread_mutex_unlock(&lock_counter_accomulatorsWB_done);
        
        //printf("All worker threads have incremented counter_local and are waiting on waitformain\n");
        
        pthread_mutex_lock(&lock_counter_accomulatorsWB_done);
        printf("Main thread is resetting counter_accomulatorsWB_done to 0\n");
        counter_accomulatorsWB_done = 0; // Reset for next iteration
        pthread_mutex_unlock(&lock_counter_accomulatorsWB_done);
        //printf("Main thread is done resetting counter_local and released mutex\n");
    }
    // Print weights
    printf("GLOBAL MULTITHREAD");
        for (int i = 1; i < num_layers; i++) { // Start from 1 since weights are between layers
            printf("Weights to Layer %d: \n", i);
            for (int j = 0; j < layers_size[i]; j++) {
                for (int k = 0; k < layers_size[i-1]; k++) {
                    printf(" W[%d][%d]: %lf ", j, k, global_weights[i][j * layers_size[i-1] + k]);
                }
                printf("\n");
            }       
        }

    printf("GLOBAL SERIAL");
        for (int i = 1; i < num_layers; i++) { // Start from 1 since weights are between layers
            printf("Weights to Layer %d: \n", i);
            for (int j = 0; j < layers_size[i]; j++) {
                for (int k = 0; k < layers_size[i-1]; k++) {
                    printf(" W[%d][%d]: %lf ", j, k, global_weights[i][j * layers_size[i-1] + k]);
                }
                printf("\n");
            }       
        }


}
    