#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

#define NUM_THREADS 10

pthread_cond_t waitforthread = PTHREAD_COND_INITIALIZER;
pthread_cond_t startwork = PTHREAD_COND_INITIALIZER; // Condition variable to signal start of work
pthread_cond_t waitformain = PTHREAD_COND_INITIALIZER; //condition variable to signal the threads that main thread is ready to make them continue
pthread_mutex_t counter_lock = PTHREAD_MUTEX_INITIALIZER;
int counter = 0;
int start_flag = 0; // Flag to indicate workers can start
int stop_flag = 0; // Flag to indicate workers should stop



// Function for each worker thread
void* worker(void *arg) {
    long thread_id = (long)arg;
    
    // Wait for the signal to start work
    pthread_mutex_lock(&counter_lock);
    while (!start_flag) { // Wait until start_flag is set
        pthread_cond_wait(&startwork, &counter_lock);
    }
    pthread_mutex_unlock(&counter_lock);
    
    while(1){
        if (stop_flag) break;
        // Begin work after being signaled
        printf("Thread %ld is working\n", thread_id); 
        sleep(0.1); // Simulate work
        printf("Thread %ld is done\n", thread_id);
        
        // Signal main thread that this worker is done
        pthread_mutex_lock(&counter_lock);
        printf("Thread %ld is incrementing counter and took mutex\n", thread_id);
        counter++;
        if (counter == NUM_THREADS){
            printf("Thread %ld is signaling waitforthread, and released mutex\n", thread_id);
            pthread_cond_signal(&waitforthread);
        }
        //wait for main thread to signal that it is ready to continue
        printf("Thread %ld is waiting on waitformain and released mutex\n", thread_id);
        pthread_cond_wait(&waitformain, &counter_lock);
        pthread_mutex_unlock(&counter_lock);
        
    }

    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    
    for (long i = 0; i < NUM_THREADS; i++) {
        printf("Creating thread %ld\n", i);
        pthread_create(&threads[i], NULL, worker, (void*)i);
    }
    
    // Main thread work before signaling workers to start
    printf("Main thread is working before waiting for workers\n");
    
    // Signal worker threads to start work
    pthread_mutex_lock(&counter_lock);
    start_flag = 1;
    pthread_cond_broadcast(&startwork);
    pthread_mutex_unlock(&counter_lock);
    
    for (int i = 0; i < 5; i++) {
        printf("\n");

        // Wait for all worker threads to signal they are done
        pthread_mutex_lock(&counter_lock);
        while (counter < NUM_THREADS){
            printf("Main thread is waiting on waitforthread\n");
            pthread_cond_wait(&waitforthread, &counter_lock);
        }
        pthread_mutex_unlock(&counter_lock);
        
        printf("All worker threads have incremented counter and are waiting on waitformain\n");
        
        pthread_mutex_lock(&counter_lock);
        printf("Main thread is resetting counter and took mutex\n");
        counter = 0; // Reset counter for next iteration
        pthread_mutex_unlock(&counter_lock);
        printf("Main thread is done resetting counter and released mutex\n");
        
        // Signal worker threads to continue
        printf("Main thread is ready to continue, broadcasting waiting working threads\n");
        if (i == 4){
            printf("Main thread is done\n");
            stop_flag = 1;
            pthread_cond_broadcast(&waitformain); // Use broadcast to signal all waiting threads
            break;
        }
        else{
            pthread_cond_broadcast(&waitformain); // Use broadcast to signal all waiting threads
        }
    }
    // Join all worker threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    

    return 0;
}


