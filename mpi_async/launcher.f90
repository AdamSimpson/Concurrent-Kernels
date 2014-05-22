program main
    use sleep
    implicit none
    include 'mpif.h'

    integer    ::  max_kernels, num_kernels, i, ierr, rank, size
    integer(8) :: cycles
    real(8)    :: start, stop, seconds

    call MPI_Init(ierr)

    ! Get number of cycles to sleep for 1 second
    seconds = 1.0
    cycles = get_cycles(seconds)

    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)

    ! Number of kernels to launch
    max_kernels = size

    ! Loop through number of kernels to launch, from 1 to max_kernels
    do num_kernels = 1, max_kernels

        ! Start timer
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        start = MPI_Wtime()

        ! Launch num_kernel kernels asynchrnously
        if (rank < num_kernels) then
            call sleep_kernel(cycles)
            call wait_for_gpu()
        endif

        ! Stop timer
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        stop = MPI_Wtime()

        ! Print seconds ellapsed
        if (rank == 0) then
            print *, 'Total time for ', num_kernels,' kernels: ', stop-start, 'seconds'
        endif

    enddo

    call MPI_Finalize(ierr)

end program main
