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
__global__ void sleep(int64_t *array, int64_t num_elems)
{
    int i;
    for(i=0; i<num_elems; i++)
        array[i] = sin((double)array[i]);
}

extern "C" void allocate_mem(int64_t **device_value, int64_t num_elems)
{
        gpuErrchk( cudaMalloc((void**)device_value, num_elems*sizeof(int64_t)) );
}

extern "C" void copy_to_device(int64_t *host_array, int64_t *device_array, int64_t num_elems)
{
    gpuErrchk( cudaMemcpy(device_array, host_array, num_elems*sizeof(int64_t), cudaMemcpyHostToDevice) );
}


extern "C" void copy_from_device(int64_t *host_array, int64_t *device_array, int64_t num_elems)
{
    gpuErrchk( cudaMemcpy(host_array, device_array, 1*sizeof(int64_t), cudaMemcpyDeviceToHost) );
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
