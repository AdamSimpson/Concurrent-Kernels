#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "mpi.h"

void sleep_kernel(int64_t *completed_cycles, int64_t requested_cycles);
void allocate_mem(int64_t **device_value, int64_t num_elems);
void copy_to_device(int64_t *host_array, int64_t *device_array, int64_t num_elems);
void copy_from_device(int64_t *host_array, int64_t *device_array, int64_t num_elems);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int64_t cycles;
    double start, stop;
    int64_t *device_cycles;
    int num_kernels, num_elems;

    num_elems = 1000000;
    int64_t *host_cycles = malloc(num_elems*sizeof(int64_t)) ;

    // Number of kernels to launch
    int max_kernels = size;

    allocate_mem(&device_cycles, num_elems);
    copy_to_device(host_cycles, device_cycles, num_elems);

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
            sleep_kernel(device_cycles, num_elems);
            copy_from_device(host_cycles, device_cycles, num_elems);
            stop = MPI_Wtime();
            printf("rank %d, cycles returned: %lld in %f s\n", rank, *host_cycles, stop-start);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
