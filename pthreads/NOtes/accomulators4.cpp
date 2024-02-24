#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include <cstdlib> // Include the <cstdlib> header to define the malloc function
#include <math.h>

#define NUM_THREADS 8000
#define REP 100

pthread_cond_t cond_waitfor_main_accomulatorWB = PTHREAD_COND_INITIALIZER;
pthread_cond_t cond_waitfor_accomulatorWB_main = PTHREAD_COND_INITIALIZER;
pthread_cond_t cond_waitfor_main_accomulatorWBpause = PTHREAD_COND_INITIALIZER;
pthread_mutex_t lock_accomulatorWB = PTHREAD_MUTEX_INITIALIZER;
pthread_barrier_t barrier_accomulatorWB;
int counter_accomulatorWB_finished = 0;
int flag_mainworking_accoumulatorWB = 0;
int flag_accomulatorWB_start = 0;
int flag_accomulatorWB_stop = 0;

void *sillyfunc(void *threadid){
    long thread_if;
    thread_if = (long)threadid;
    printf("Hello World! It's me, thread #%ld!\n", thread_if);


    while(1){
        // pthread_barrier_wait(&barrier_accomulatorWB);
        pthread_mutex_lock(&lock_accomulatorWB);
        while (flag_accomulatorWB_start == 0){
            printf("Thread #%ld waiting on cond_waitfor_main_accomulatorWB because flag_accomulatorWB_start = 0\n", thread_if);
            pthread_cond_wait(&cond_waitfor_main_accomulatorWB, &lock_accomulatorWB);
        }
        pthread_mutex_unlock(&lock_accomulatorWB);
        if (flag_accomulatorWB_stop == 1) break;
        printf("Thread #%ld working\n", thread_if);
        sleep(0);
        printf("Thread #%ld finished\n", thread_if);
        //------pause main thread until all threads finish
        pthread_mutex_lock(&lock_accomulatorWB);
        counter_accomulatorWB_finished++;//incrase the counter and in case of the last thread, signal the main
        if (counter_accomulatorWB_finished == NUM_THREADS) {pthread_cond_signal(&cond_waitfor_accomulatorWB_main); printf("Thread #%ld signals main\n", thread_if);}
        printf("thread #%d waits on cond_waitfor_main_accomulatorWBpause because increases counter to %d\n", thread_if, counter_accomulatorWB_finished);
        pthread_cond_wait(&cond_waitfor_main_accomulatorWBpause, &lock_accomulatorWB);
        pthread_mutex_unlock(&lock_accomulatorWB);
    }

    return NULL;
}

int main(){

    int i = 0;
    printf("i = %d\n", i);
    pthread_t threads[NUM_THREADS];
    for (i=0; i<NUM_THREADS; i++){
        pthread_create(&threads[i], NULL, sillyfunc, (void *)i);
    }
    // pthread_barrier_init(&barrier_accomulatorWB,NULL,NUM_THREADS);
    flag_mainworking_accoumulatorWB = 1;
    i = 0;
    while (i<REP){
        //-------------------work
        printf("-\n-\n-\n-\n");
        printf("Main working\n");
        printf("-\n-\n-\n-\n"); 
        sleep(0);
        //------------------------------------------------
        pthread_mutex_lock(&lock_accomulatorWB);
        //start threads
        printf("main set flag_accomulatorWB_start to 1\n");
        flag_accomulatorWB_start = 1;
        printf("main sets flag_mainworking_accoumulatorWB to 0\n");
        printf("main signal threads waiting on cond_waitfor_main_accomulatorWB\n");
        flag_mainworking_accoumulatorWB = 0;
        pthread_cond_broadcast(&cond_waitfor_main_accomulatorWB);
        pthread_mutex_unlock(&lock_accomulatorWB);

        //------pause main thread until all threads finish
        pthread_mutex_lock(&lock_accomulatorWB);
        while (counter_accomulatorWB_finished < NUM_THREADS){
            printf("Main waiting on cond_waitfor_accomulatorWB_main because counter_accomulatorWB_finished = %d\n", counter_accomulatorWB_finished);
            pthread_cond_wait(&cond_waitfor_accomulatorWB_main, &lock_accomulatorWB);
        }
        pthread_mutex_unlock(&lock_accomulatorWB);
        pthread_mutex_lock(&lock_accomulatorWB);
        printf("all threads finished and waiting on cond_waitfor_main_accomulatorWBpause\n");
        printf("Main resumes because all threads finished\n");
        printf("main sets flag_mainworking_accoumulatorWB to 1\n");
        printf("main sets flag_accomulatorWB_start to 0\n");
        printf("main sets counter_accomulatorWB_finished to 0\n");
        counter_accomulatorWB_finished = 0;//resetting the counter of finished accomulators
        flag_accomulatorWB_start = 0; // pause accomulatorsWB 
        flag_mainworking_accoumulatorWB = 1;
        printf("main thread signals threads waiting on cond_waitfor_main_accomulatorWBpause\n");
        pthread_cond_broadcast(&cond_waitfor_main_accomulatorWBpause);
        pthread_mutex_unlock(&lock_accomulatorWB);
        
        
        i++;
        if (i == REP) {
            printf("Main finished\n"); 
            pthread_mutex_lock(&lock_accomulatorWB);
            flag_accomulatorWB_stop = 1;
            flag_accomulatorWB_start = 1; //to make htem go out of the loop
            pthread_cond_broadcast(&cond_waitfor_main_accomulatorWB);
            pthread_mutex_unlock(&lock_accomulatorWB);
            break;}
    }
    printf("Main finished\n");
    for (i=0; i<NUM_THREADS; i++){
    pthread_join(threads[i], NULL);
    }
    return 0;
}