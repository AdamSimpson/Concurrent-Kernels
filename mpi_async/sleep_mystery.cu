#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// CUDA kernel to pause for at least num_cycle cycles
__global__ void sleep(int64_t *completed_cycles, int64_t requested_cycles)
{
    completed_cycles[0] = 0;
    int64_t start = clock64();
    while(completed_cycles[0] < requested_cycles) {
        completed_cycles[0] = clock64() - start;
    }
}

extern "C" void allocate_mem(int64_t **device_value)
{
        gpuErrchk( cudaMalloc((void**)device_value, sizeof(int64_t)) );
}

extern "C" void copy_mem(int64_t *host_value, int64_t *device_value)
{
    gpuErrchk( cudaMemcpy(host_value, device_value, sizeof(int64_t), cudaMemcpyDeviceToHost) );
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
extern "C" void sleep_kernel(int64_t *completed_cycles, int64_t requested_cycles)
{
    // Our kernel will launch a single thread to sleep the kernel
    int blockSize, gridSize;
    blockSize = 1;
    gridSize = 1;

    // Execute the kernel in default stream
    sleep<<< gridSize, blockSize >>>(completed_cycles, requested_cycles);
}

// Wait for all work  to complete
extern "C" void wait_for_gpu()
{
    cudaDeviceSynchronize();
}
