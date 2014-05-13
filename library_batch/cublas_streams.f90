program main
    use ISO_C_BINDING
    use cublas_f
    implicit none

    integer :: dim, stat, i, j, k, batch_count
    integer(8) :: bytes
    real(8),dimension(:,:),allocatable :: A, B, C
    real(8) :: alpha, beta, index, sum
    type(C_PTR), dimensions(:), allocatable :: d_A, d_B, d_C, streams
    type(C_PTR) :: handle

    integer :: sizeof_double
    parameter (sizeof_double=8)

    !Linear dimension of matrices
    dim = 100

    ! Number of A,B,C matrix sets
    batch_count = 1000

    ! Allocate host storage for A,B,C square matrices
    allocate(A(dim,dim,batch_count))
    allocate(B(dim,dim,batch_count))
    allocate(C(dim,dim,batch_count))

    ! Allocate device storage for A,B,C
    allocate(d_A(batch_count))
    allocate(d_B(batch_count))
    allocate(d_C(batch_count))
    bytes = int(dim*dim*sizeof_double, 8)
    do i=1,batch_count
        stat = cudaMalloc(d_A(i), bytes)
        stat = cudaMalloc(d_B(i), bytes)
        stat = cudaMalloc(d_C(i), bytes)
    enddo

    ! Fill A,B diagonals with sin(i) data, C diagonal with cos(i)^2
    ! Matrices are arranged column major
    do k=1,batch_count
        do j=1,dim
            do i=1,dim
                if (i==j) then
                    index = real(j*dim + i)
                    A(i,j,k) = k*sin(index)
                    B(i,j,k) = sin(index)
                    C(i,j,k) = k*cos(index) * cos(index)
                else
                    A(i,j,k) = 0.0
                    B(i,j,k) = 0.0
                    C(i,j,k) = 0.0
                endif
            enddo ! i
        enddo ! j
    enddo ! k

    ! Create cublas instance
    stat = cublasCreate(handle)

    ! Set input matrices on device
    do i=1,batch_count
        stat = cublasSetMatrix(dim, dim, sizeof_double, C_LOC(A(1,1,i)), dim, d_A(i), dim)
        stat = cublasSetMatrix(dim, dim, sizeof_double, C_LOC(B(1,1,i)), dim, d_B(i), dim)
        stat = cublasSetMatrix(dim, dim, sizeof_double, C_LOC(C(1,1,i)), dim, d_C(i), dim)
    enddo

    ! Create a stream for every DGEMM operation
    allocate(streams(batch_count))
    do i=1,batch_count
        stat = cublasStreamCreate(streams(i))
    enddo

    ! DGEMM: C = alpha*A*B + beta*C
    alpha = 1.0
    beta = 1.0
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, &
                       dim, dim, dim, &
                       alpha,         &
                       d_A, dim,      &
                       d_B, dim,      &
                       beta,          &
                       d_C, dim)

    ! Retrieve result matrix from device
    stat = cublasGetMatrix(dim, dim, sizeof_double, d_C, dim, C_LOC(C(1,1)), dim)

    ! Simple sanity test, sum up all elements
    sum = 0.0
    do j=1,dim
        do i=1,dim
            sum = sum + C(i,j)
        enddo
    enddo
    print *, "Sum is:", sum, "should be: ", dim

    deallocate(A)
    deallocate(B)
    deallocate(C)
    call cudaFree(d_A)
    call cudaFree(d_B)
    call cudaFree(d_C)
    call cublasDestroy(handle)

end program
