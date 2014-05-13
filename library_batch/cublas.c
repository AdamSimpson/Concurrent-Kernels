#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include "cublas_v2.h"

int main(int argc, char* argv[])
{
    // Linear dimension of matrices
    int dim = 100;

    // Allocate host storage for A,B,C square matrices
    double *A, *B, *C;
    A = (double*)malloc(dim*dim*sizeof(double));
    B = (double*)malloc(dim*dim*sizeof(double));
    C = (double*)malloc(dim*dim*sizeof(double));

    // Allocate device storage for A,B,C
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, dim*dim*sizeof(double));
    cudaMalloc((void**)&d_B, dim*dim*sizeof(double));
    cudaMalloc((void**)&d_C, dim*dim*sizeof(double));

    // Fill A,B diagonals with sin(i) data, C diagonal with cos(i)^2
    // Matrices are arranged column major
    int i,j,index;
    for(j=0; j<dim; j++) {
        for(i=0; i<dim; i++) {
            index = j*dim + i;
            if(i==j) {
                A[index] = sin(index);
                B[index] = sin(index);
                C[index] = cos(index)*cos(index);
            }
	    else {
                A[index] = 0.0;
                B[index] = 0.0;
                C[index] = 0.0;
            }
        }    
    }

    // Create cublas instance
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set input matrices on device
    cublasSetMatrix(dim, dim, sizeof(double), A, dim, d_A, dim);
    cublasSetMatrix(dim, dim, sizeof(double), B, dim, d_B, dim);
    cublasSetMatrix(dim, dim, sizeof(double), C, dim, d_C, dim);

    // DGEMM: C = alpha*A*B + beta*C
    double alpha = 1.0;
    double beta  = 1.0;

    cublasDgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                dim, dim, dim,
                &alpha,
                d_A, dim,
                d_B, dim,
                &beta,
                d_C, dim);

    // Retrieve result matrix from device
    cublasGetMatrix(dim, dim, sizeof(double), d_C, dim, C, dim);

    // Simple sanity test, sum up all elements
    double sum = 0;
    for(j=0; j<dim; j++) {
        for(i=0; i<dim; i++) {
            index = j*dim + i;
            sum += C[index];
        }
    }
    printf("Sum is: %f, should be: %d\n", sum, dim);   

    // Clean up resources
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}
