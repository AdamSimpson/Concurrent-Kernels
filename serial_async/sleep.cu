#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static cudaStream_t *streams;

// CUDA kernel to pause for at least num_cycle cycles
__global__ void sleep(int64_t num_cycles)
{
    int64_t cycles = 0;
    int64_t start = clock64();
    while(cycles < num_cycles) {
        cycles = clock64() - start;
    }
}

// Returns number of cycles required for requested seconds
extern "C" int64_t get_cycles(float seconds)
{
    // Get device frequency in KHz
    int64_t Hz;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    Hz = int64_t(prop.clockRate) * 1000;

    // Calculate number of cycles to wait
    int64_t num_cycles;
    num_cycles = (int64_t)(seconds * Hz);
  
    return num_cycles;
}

// Create streams
extern "C" void create_streams(int num_streams)
{
    // Allocate streams
    streams = (cudaStream_t *) malloc(num_streams*sizeof(cudaStream_t));    

    // Create streams
    int i;
    for(i = 0; i < num_streams; i++)
        cudaStreamCreate(&streams[i]);
}

// Launches a kernel that sleeps for num_cycles
extern "C" void sleep_kernel(int64_t num_cycles, int stream_id)
{
    // Our kernel will launch a single thread to sleep the kernel
    int blockSize, gridSize;
    blockSize = 1;
    gridSize = 1;

    // Execute the kernel
    sleep<<< gridSize, blockSize, 0, streams[stream_id] >>>(num_cycles);
}

// Wait for streams to complete and then destroy
extern "C" void destroy_streams(int num_streams)
{
    int i;
    for(i = 0; i < num_streams; i++)
    {
        // Wait for kernel to finish
        cudaStreamSynchronize(streams[i]);

        // Clean up stream
        cudaStreamDestroy(streams[i]);
    }

    free(streams);
}
