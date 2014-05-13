program main
    use ISO_C_BINDING
    use cublas_f
    implicit none

    integer :: dim, stat, i, j
    integer(8) :: bytes
    real(8),dimension(:,:),allocatable  :: A, B, C
    real(8) :: alpha, beta, index, sum
    type(C_PTR) :: d_A, d_B, d_C
    type(C_PTR) :: handle

    integer :: sizeof_double
    parameter (sizeof_double=8)

    !Linear dimension of matrices
    dim = 100

    ! Allocate host storage for A,B,C square matrices
    allocate(A(dim,dim))
    allocate(B(dim,dim))
    allocate(C(dim,dim))

    ! Allocate device storage for A,B,C
    bytes = int(dim*dim*sizeof_double, 8) 
    stat = cudaMalloc(d_A, bytes)
    stat = cudaMalloc(d_B, bytes)
    stat = cudaMalloc(d_C, bytes)

    ! Fill A,B diagonals with sin(i) data, C diagonal with cos(i)^2
    ! Matrices are arranged column major
    do j=1,dim
        do i=1,dim
            if (i==j) then
                index = real(j*dim + i)
                A(i,j) = sin(index)
                B(i,j) = sin(index)
                C(i,j) = cos(index) * cos(index)
            else
                A(i,j) = 0.0
                B(i,j) = 0.0
                C(i,j) = 0.0
            endif
        enddo
    enddo

    ! Create cublas instance
    stat = cublasCreate(handle)

    ! Set input matrices on device
    stat = cublasSetMatrix(dim, dim, sizeof_double, C_LOC(A(1,1)), dim, d_A, dim)
    stat = cublasSetMatrix(dim, dim, sizeof_double, C_LOC(B(1,1)), dim, d_B, dim)
    stat = cublasSetMatrix(dim, dim, sizeof_double, C_LOC(C(1,1)), dim, d_C, dim)

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
