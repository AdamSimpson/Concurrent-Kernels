program main
    use sleep

    implicit none

    include 'mpif.h'

    integer(8) :: cycles
    integer    ::  max_kernels, num_kernels, i, ierr, rank, size
    real(8)    :: start, stop, seconds

    call MPI_Init()

    ! Get number of cycles to sleep for 1 second
    seconds = 1.0
    cycles = get_cycles(seconds)

    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)

    do num_kernels = 1, max_kernels

        ! Start timer
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        start = MPI_Wtime()

        ! Create streams
        call create_streams(num_kernels)

        ! Launch num_kernel kernels asynchrnously
        if (rank < num_kernels) then
            call sleep_kernel(cycles, rank+1)
        endif

        ! Wait for kernels to complete and clean up streams
        call destroy_streams(num_kernels)

        ! Stop timer
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        stop = MPI_Wtime()

        if (rank == 0) then
            print *, 'Total time for ', num_kernels,' kernels: ', stop-start, 'seconds'
        endif

    enddo

    call MPI_Finalize()

end program main
