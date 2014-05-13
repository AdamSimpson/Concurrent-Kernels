#include <stdio.h>
#include "cublas_v2.h"

extern "C" int f_cublasCreate(cublasHandle_t **handle)
{
    *handle = (cublasHandle_t*)malloc(sizeof(cublasHandle_t));
    return cublasCreate(*handle);
}

extern "C" int f_cublasDgemm(cublasHandle_t *handle,
               cublasOperation_t transa, cublasOperation_t transb, 
              int m, int n, int k, 
              const double *alpha,
              const double *A, int lda, 
              const double *B, int ldb,
              const double *beta, 
              double *C, int ldc)
{
    return cublasDgemm(*handle,transa,transb,m,n,k,alpha,A,lda,B,ldb,beta,C,ldc);
}

extern "C" void f_cublasDestroy(cublasHandle_t *handle_ptr)
{
    cublasDestroy(*handle_ptr);
    free(handle_ptr);
}

extern "C" int f_cudaStreamCreate(cudaStream_t **stream)
{
    *stream = (cudaStream_t *) malloc(sizeof(cudaStream_t));
    cudaStreamCreate(*stream);
}

