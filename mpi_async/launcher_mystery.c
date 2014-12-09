#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "mpi.h"

int64_t get_cycles(float seconds);
void sleep_kernel(int64_t *completed_cycles, int64_t requested_cycles);
void allocate_mem(int64_t **device_cycles);
void copy_mem(int64_t *host_cycles, int64_t *device_cycles);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int64_t cycles;
    double start, stop;
    int64_t host_cycles=-1;
    int64_t *device_cycles;
    int num_kernels;

    // Get number of cycles to sleep for 1 second
    cycles = get_cycles(1.0);

    // Number of kernels to launch
    int max_kernels = size;

    allocate_mem(&device_cycles);

    // Loop through number of kernels to launch, from 1 to max_kernels
    for(num_kernels=1; num_kernels<=max_kernels; num_kernels++)
    {
        // Start timer
        if(rank == 0)
            printf("\n\nStarting kernels %d\n", num_kernels);
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();

        // Launch kernel asynchrnously
        if(rank < num_kernels) {
            start = MPI_Wtime();
            sleep_kernel(device_cycles, cycles);
            copy_mem(&host_cycles, device_cycles);
            stop = MPI_Wtime();
            printf("rank %d, cycles returned: %lld in %f s\n", rank, host_cycles, stop-start);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
