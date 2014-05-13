#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "mpi.h"

int64_t get_cycles(float seconds);
void sleep_kernel(int64_t num_cycles, int stream_id);
void create_streams(int num_streams);
void destroy_streams(int num_streams);

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    uint64_t cycles;
    double start, stop;
    int i, num_kernels;

    // Get number of cycles to sleep for 1 second
    cycles = get_cycles(1.0);

    // Number of kernels to launch
    int max_kernels = 33;

    // Loop through number of kernels to launch, from 1 to num_kernels
    for(num_kernels=1; num_kernels<=max_kernels; num_kernels++)
    {
        // Start timer
        MPI_Barrier(MPI_COMM_WORLD);
        start = MPI_Wtime();

        // Create streams
        create_streams(num_kernels);

        // Launch kernel asynchrnously
        if(rank < num_kernels)
            sleep_kernel(cycles, i);

        // Wait for kernels to complete and clean up streams
        destroy_streams(num_kernels);

        // Stop timer
        MPI_Barrier(MPI_COMM_WORLD);
        stop = MPI_Wtime();

        // Print seconds ellapsed
        if(rank == 0)
            printf("elapsed time %f seconds\n", stop-start);

    }

    MPI_Finalize();

    return 0;
}
