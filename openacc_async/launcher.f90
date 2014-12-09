program main
    implicit none

    integer(8) :: num_cycles, cycles, foo_cycles, i
    integer    :: max_kernels, num_kernels, stream_id
    real(8)    :: start, stop, seconds

    ! Set number of cycles to sleep for 1 second
    ! We'll use frequency/instruction latency to estimate this
    num_cycles = 730000000/15

    ! Maximum number of kernels to launch
    max_kernels = 33

    ! Loop through number of kernels to launch, from 1 to num_kernels
    do num_kernels = 1, max_kernels

        ! Start timer
        call cpu_time(start)

        ! Launch num_kernel kernels asynchrnously
        do i = 1, num_kernels
            !$acc parallel async(i) vector_length(1) num_gangs(1), private(cycles, foo_cycles), firstprivate(num_cycles)
            !$acc loop seq
            do cycles = 1,num_cycles
                foo_cycles = cycles
            enddo
            !$acc end loop
            !$acc end parallel
        enddo 

        ! Wait for all async streams to complete
        !$acc wait

        ! Stop timer
        call cpu_time(stop)

        print *, 'Total time for ', num_kernels,' kernels: ', stop-start, 'seconds'

    enddo

end program main
