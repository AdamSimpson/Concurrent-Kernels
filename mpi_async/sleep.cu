#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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

// Launches a kernel that sleeps for num_cycles
extern "C" void sleep_kernel(int64_t num_cycles)
{
    // Our kernel will launch a single thread to sleep the kernel
    int blockSize, gridSize;
    blockSize = 1;
    gridSize = 1;

    // Execute the kernel in default stream
    sleep<<< gridSize, blockSize >>>(num_cycles);
}

// Wait for all work  to complete
extern "C" void wait_for_gpu()
{
    cudaDeviceSynchronize();
}
