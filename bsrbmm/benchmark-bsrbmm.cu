#include <iostream>
#include <sys/time.h>

#define TEST_TIMES 1 // do not test more than 1 time for now (vector not clean)
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#include "mmio_highlevel.h"
#include "csr2bsr_batch.cu"

bool check_result(float* p1, float* p2, const int N)
{
    bool flag = true;
    for (int i = 0; i < N; i ++) {

        float diff = p1[i] - p2[i];
        if (fabs(diff) > 1e-6) {
            printf("[%d](%.f,%.f),", i, p1[i], p2[i]);
            flag = false;
        }
    }
    return flag;
}

bool check_result(float* p1, int* p2, const int N)
{
    bool flag = true;
    for (int i = 0; i < N * N; i ++) {
        //printf("(%.0f,%d)",p1[i],p2[i]);
        float diff = p1[i] - (float)p2[i];
        if (fabs(diff) > 1e-6) {
            flag = false;
        }
    }
    return flag;
}

int countnnzinvec(const float* vec, const int N)
{
    int counter = 0;
    for (int i=0; i<N; i++) if (vec[i] != 0) counter += 1;
    return counter;
}

void printvec(float* vec, const int N)
{
    for(int i=0; i<N; i++) printf(vec[i]>0?"1":"0");
    printf("\n");
}

void printresvec(float* vec, const int N)
{
    for(int i=0; i<N; i++) printf("%d", (int)vec[i]);
    printf("\n");
}

void printmat(float* bsrval, const int nblocks, const int blocksize)
{
    for(int i=0; i<nblocks; i++) {
        printf("[%d]\n", i);
        for(int j=0; j<blocksize; j++) {
            for(int k=0; k<blocksize; k++) {
                printf(bsrval[i*blocksize*blocksize+j*blocksize+k]>0?"1":"0");
            }
            printf("\n");
        }
        printf("\n");
    }
}

void printbinvec(unsigned* binvec, const int N)
{
    for(int i=0; i<N; i++) bin(binvec[i]);
    printf("\n");
}

void printbinmat(unsigned* binbsrval, const int nblocks, const int blocksize)
{
    for(int i=0; i<nblocks; i++) {
        printf("[%d]\n", i);
        for(int j=0; j<blocksize; j++) {
            bin(binbsrval[i*blocksize+j]);
            printf("\n");
        }
        printf("\n");
    }
}

__global__ void printpackvec (ullong* packvec, const int N)
{
    for(int i=0; i<N; i++) {
        ullong j;
        for(j = 1ULL << 63; j > 0; j = j / 2)
            (packvec[i] & j) ? printf("1") : printf("0");
    }
    printf("\n");
}

__global__ void printpackmat (ullong* packbsrval,  const int nblocks, const int blocksize)
{
    for(int i=0; i<nblocks; i++) {
        printf("[%d]\n", i);
        for(int j=0; j<blocksize; j++) {
            ullong k;
            for(k = 1ULL << 63; k > 0; k = k / 2)
                (packbsrval[i*blocksize+j] & k) ? printf("1") : printf("0");

            printf("\n");
        }
        printf("\n");
    }
}

void printind(int* indarr, const int N)
{
    for(int i=0; i<N; i++) {
        printf("%d ", indarr[i]);
    }
    printf("\n");
}

int main32(int argc, char* argv[])
{

    cudaSetDevice(0);
    if (argc < 2)
    {
        printf("./exe [xxx.mtx]\n");
        exit(1);
    }

    // matrix storage -----------------------------------
    // read sparse matrix from file and store as csr format
    // matrix metadata
    char *filename = argv[1]; // e.g. "G43.mtx"
    printf("input sparse matrix: %s\n", filename);

    int nrows, ncols, nnz, isSymmetric;
    mmio_info<float>(&nrows, &ncols, &nnz, &isSymmetric, filename);
    printf("nrows: %d, ncols: %d, nnz: %d, isSymmetric: ", nrows, ncols, nnz); printf(isSymmetric?"true\n":"false\n");

    // matrix csr in host
    int* h_csrRowPtr, *h_csrColInd;
    float* h_csrVal;
    h_csrRowPtr = (int*) malloc(sizeof(int) * (nrows+1));
    h_csrColInd = (int*) malloc(sizeof(int) * nnz);
    h_csrVal = (float*) malloc(sizeof(float) * nnz);
    mmio_data<float>(h_csrRowPtr, h_csrColInd, h_csrVal, filename);

    // copy csr to device
    int* csrRowPtr, *csrColInd;
    float* csrVal;
    cudaMalloc(&csrRowPtr, sizeof(int) * (nrows+1));
    cudaMalloc(&csrColInd, sizeof(int) * nnz);
    cudaMalloc(&csrVal, sizeof(float) * nnz);
    cudaMemcpy(csrRowPtr, h_csrRowPtr, sizeof(int) * (nrows+1), cudaMemcpyHostToDevice);
    cudaMemcpy(csrColInd, h_csrColInd, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(csrVal, h_csrVal, sizeof(float) * nnz, cudaMemcpyHostToDevice);

    // process input matrix to simulate tc algorithm
    // C = A * A^T
    // 1) get A = graphblas::tril(A)
    // duplicate matrix as A
    int* A_csrRowPtr, *A_csrColInd;
    float* A_csrVal;
    int* d_A_nnz;
    int A_nnz;
    cudaMalloc(&A_csrRowPtr, sizeof(int) * (nrows+1));
    cudaMalloc(&A_csrColInd, sizeof(int) * nnz);
    cudaMalloc(&A_csrVal, sizeof(float) * nnz);
    cudaMalloc(&d_A_nnz, sizeof(int) * 1);
    cudaMemcpy(A_csrRowPtr, csrRowPtr, sizeof(int) * (nrows+1), cudaMemcpyDeviceToDevice);
    cudaMemcpy(A_csrColInd, csrColInd, sizeof(int) * nnz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(A_csrVal, csrVal, sizeof(float) * nnz, cudaMemcpyDeviceToDevice);

    // call tril()
    tril_csr<int, float><<<1,1>>>(csrRowPtr, csrColInd, csrVal, nrows, nnz,
                                  A_csrRowPtr, A_csrColInd, A_csrVal, d_A_nnz);
    cudaMemcpy(&A_nnz, d_A_nnz, sizeof(int) * 1, cudaMemcpyDeviceToHost);
    printf("nnz before tril_csr(): %d, after tril_csr(): %d\n", nnz, A_nnz); // <- we ignore A's [A_nnz to nnz] from now

    // reset host csr with updated matrix
    free(h_csrVal);
    free(h_csrColInd);
    free(h_csrRowPtr);
    h_csrRowPtr = (int*) malloc(sizeof(int) * (nrows+1));
    h_csrColInd = (int*) malloc(sizeof(int) * A_nnz);
    h_csrVal = (float*) malloc(sizeof(float) * A_nnz);
    cudaMemcpy(h_csrRowPtr, A_csrRowPtr, sizeof(int) * (nrows+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csrColInd, A_csrColInd, sizeof(int) * A_nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csrVal, A_csrVal, sizeof(float) * A_nnz, cudaMemcpyDeviceToHost);

	// transform from csr to bsr using cuSPARSE
	int* A_bsrRowPtr, *A_bsrColInd;
	float* A_bsrVal;
	int blocksize = 32;
	int mb = (nrows + blocksize-1)/blocksize;
    int nb = (ncols + blocksize-1)/blocksize;
    int nblockrows = mb;

	// cuSPARSE API metadata setup
    cusparseMatDescr_t csr_descr = 0;
    cusparseCreateMatDescr(&csr_descr);
    cusparseSetMatType(csr_descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(csr_descr,CUSPARSE_INDEX_BASE_ZERO);
    cusparseMatDescr_t bsr_descr = 0;
    cusparseCreateMatDescr(&bsr_descr);
    cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO);
    cudaStream_t streamId = 0;
    cusparseHandle_t handle = 0;
    cusparseCreate(&handle);
    cusparseSetStream(handle, streamId);
    cusparseDirection_t dirA = CUSPARSE_DIRECTION_COLUMN;

    // csr2bsr in column-major order, estimate first
    int nblocks;

    cudaMalloc((void**)&A_bsrRowPtr, sizeof(int) *(nblockrows+1));
    cusparseXcsr2bsrNnz(handle, dirA, nrows, ncols, csr_descr,
                        A_csrRowPtr, A_csrColInd, blocksize, bsr_descr, A_bsrRowPtr, &nblocks);
    cudaMalloc((void**)&A_bsrColInd, sizeof(int)*nblocks);
    printf("blocksize: %d, nblockrows: %d, nblocks: %d\n", blocksize, nblockrows, nblocks);

    cusparseDestroyMatDescr(csr_descr);
    cusparseDestroyMatDescr(bsr_descr);
    cusparseDestroy(handle);

    // packed matrix tA
    unsigned* tA;
    cudaMalloc((void**)&tA, nblocks * blocksize * sizeof(unsigned));
    csr2bsr_batch_32(h_csrRowPtr, h_csrColInd, nrows, ncols, A_nnz,
                     A_bsrRowPtr, A_bsrColInd, tA, blocksize, nblockrows, nblocks);

    // csr2csc for B as A^T
    int* B_cscRowInd, *B_cscColPtr;
    float* dummy_B_cscVal;
    cudaMalloc(&B_cscRowInd, sizeof(int) * A_nnz);
    cudaMalloc(&B_cscColPtr, sizeof(int) * (nrows+1));

    cusparseHandle_t handle_csr2csc;
    cusparseCreate(&handle_csr2csc);
    cusparseScsr2csc(handle_csr2csc, nrows, ncols, A_nnz,
                     A_csrVal, A_csrRowPtr, A_csrColInd,
                     dummy_B_cscVal, B_cscRowInd, B_cscColPtr,
                     CUSPARSE_ACTION_SYMBOLIC,
                     CUSPARSE_INDEX_BASE_ZERO);
    cusparseDestroy(handle_csr2csc);

    int *h_B_cscRowInd, *h_B_cscColPtr;
    h_B_cscRowInd = (int*) malloc(sizeof(int) * A_nnz);
    h_B_cscColPtr = (int*) malloc(sizeof(int) * (nrows+1));
    cudaMemcpy(h_B_cscRowInd, B_cscRowInd, sizeof(int) * A_nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B_cscColPtr, B_cscColPtr, sizeof(int) * (nrows+1), cudaMemcpyDeviceToHost);
    cudaFree(B_cscRowInd);
    cudaFree(B_cscColPtr);

    // csr2bsr for B & pack matrix for tB
    int* B_bsrRowPtr, *B_bsrColInd;
    cudaMalloc(&B_bsrRowPtr, sizeof(int) * (nblockrows+1));
    cudaMalloc(&B_bsrColInd, sizeof(int) * nblocks);
    unsigned* tB;
    cudaMalloc((void**)&tB, nblocks * blocksize * sizeof(unsigned));
    csr2bsr_batch_32(h_B_cscColPtr, h_B_cscRowInd, nrows, ncols, A_nnz,
                     B_bsrRowPtr, B_bsrColInd, tB, blocksize, nblockrows, nblocks);
    free(h_B_cscRowInd);
    free(h_B_cscColPtr);

//    printDeviceIndArr<<<1,1>>>(A_bsrRowPtr, (nblockrows+1));
//    printDeviceIndArr<<<1,1>>>(A_bsrColInd, nblocks);
//    printDeviceIndArr<<<1,1>>>(B_bsrRowPtr, (nblockrows+1));
//    printDeviceIndArr<<<1,1>>>(B_bsrColInd, nblocks);

	// time measurement setup -----------------------------------
	cudaEvent_t start, stop;
	float milliseconds = 0;

    // ============================================= BSTC-32 bsr bmm
    // allocate bsr storage for resulting C
    // use 1 float to store the reduced sum for now
    int* fC;
	cudaMalloc((void**)&fC, sizeof(int) * nblockrows);
	setDeviceValArr<<<1,1>>>(fC, nblockrows, 0);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // get grid dim
    double nbr = cbrt((double)nblockrows);
    int blockdim = (int)ceil(nbr);
    printf("cbrt(nblockrows) = %d\n", blockdim);
    dim3 grid(blockdim, blockdim, blockdim);

    // ------
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++) { // follow warp consolidation model (32 threads per block)

        bmm32_sparse<int, int><<<grid, 32>>>(tA, tB, fC,
                                               A_bsrRowPtr, A_bsrColInd,
                                               B_bsrRowPtr, B_bsrColInd,
                                               nblockrows, nblocks);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double bmm32_time = (milliseconds*1e3)/double(TEST_TIMES);

    // ------

//    printf("fC: \n"); printDeviceIndArr<<<1,1>>>(fC, nblockrows);
    int* result_bsrbmm32;
    cudaMalloc((void**)&result_bsrbmm32, sizeof(int) * 1);
    reuduceSum<<<1,1>>>(fC, nblockrows, result_bsrbmm32);
    int ntris;
    cudaMemcpy(&ntris, result_bsrbmm32, sizeof(int) * 1, cudaMemcpyDeviceToHost);
    printf("ntris: %d\n", ntris);
    cudaFree(result_bsrbmm32);


    // ============================================= cuSPARSE csr spmm-float
//    // setup cusparse metadata
//    cusparseHandle_t handle_csr;
//    cusparseCreate(&handle_csr);
//
//    cusparseMatDescr_t A_descr;
//    cusparseCreateMatDescr(&A_descr);
//    cusparseSetMatType(A_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
//    cusparseSetMatIndexBase(A_descr, CUSPARSE_INDEX_BASE_ZERO);
//
//    cusparseMatDescr_t B_descr;
//    cusparseCreateMatDescr(&B_descr);
//    cusparseSetMatType(B_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
//    cusparseSetMatIndexBase(B_descr, CUSPARSE_INDEX_BASE_ZERO);
//
//    cusparseMatDescr_t C_descr;
//    cusparseCreateMatDescr(&C_descr);
//    cusparseSetMatType(C_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
//    cusparseSetMatIndexBase(C_descr, CUSPARSE_INDEX_BASE_ZERO);
//
//    // setup input and output csr storage
//    // 2) get B = A^T (get B = A here, and transpose using CUSPARSE_OPERATION_TRANSPOSE)
//    // duplicate A as B
//    int* B_csrRowPtr, *B_csrColInd;
//    float* B_csrVal;
//    int B_nnz = A_nnz;
//    cudaMalloc(&B_csrRowPtr, sizeof(int) * (nrows+1));
//    cudaMalloc(&B_csrColInd, sizeof(int) * B_nnz);
//    cudaMalloc(&B_csrVal, sizeof(float) * B_nnz);
//    cudaMemcpy(B_csrRowPtr, A_csrRowPtr, sizeof(int) * (nrows+1), cudaMemcpyDeviceToDevice);
//    cudaMemcpy(B_csrColInd, A_csrColInd, sizeof(int) * B_nnz, cudaMemcpyDeviceToDevice);
//    cudaMemcpy(B_csrVal, A_csrVal, sizeof(float) * B_nnz, cudaMemcpyDeviceToDevice);
//
//    // calculate nnz in C and allocate storage
//    int* C_csrRowPtr, *C_csrColInd;
//    float* C_csrVal;
//    int C_nnz;
//    cudaMalloc(&C_csrRowPtr, sizeof(int) * (nrows+1));
//    cusparseXcsrgemmNnz(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, nrows, nrows, nrows,
//                        A_descr, A_nnz, A_csrRowPtr, A_csrColInd,
//                        B_descr, B_nnz, B_csrRowPtr, B_csrColInd,
//                        C_descr, C_csrRowPtr, &C_nnz);
//    cudaMalloc(&C_csrColInd, sizeof(int) * C_nnz);
//    cudaMalloc(&C_csrVal, sizeof(float) * C_nnz);
//    printf("result C nnz: %d\n", C_nnz);
//
//    //
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//
//    // ------
//    cudaEventRecord(start);
//    for (int i=0; i<TEST_TIMES; i++) {
//        cusparseScsrgemm(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, nrows, nrows, nrows,
//                         A_descr, A_nnz, A_csrVal, A_csrRowPtr, A_csrColInd,
//                         B_descr, B_nnz, B_csrVal, B_csrRowPtr, B_csrColInd,
//                         C_descr, C_csrVal, C_csrRowPtr, C_csrColInd);
//    }
//
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//
//    milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds,start,stop);
//    double cusparsecsrspgemmfloat_time = (milliseconds*1e3)/double(TEST_TIMES);
//    // ------
//
//    // the result include csrvalC, C_csrRowPtr, C_csrColInd
//    //float* result_cusparsecsrspgemmfloat = (float*)malloc(ncols * 1 * sizeof(float));
//    //cudaMemcpy(result_cusparsecsrspgemmfloat, dY, ncols * 1 * sizeof(float), cudaMemcpyDeviceToHost);
//

    //============================================= CHECK RESULT
    //printf("CuSPARSE CSR SpGEMM-float (baseline) success: %d\n", check_result(result_cusparsebsrspmvfloat, result_cusparsebsrspmvfloat, ncols));
    //printf("BSR BMM-32 success: %d\n", check_result(result_bsrbmv32, result_cusparsebsrspmvfloat, ncols));

    printf("BSR BMM-32: %.3lf\n", bmm32_time);
//    printf("CuSPARSE CSR SpGEMM-float: %.3lf\n", cusparsecsrspgemmfloat_time);

    // free bsr bmm
    cudaFree(fC);

    cudaFree(tB);
    cudaFree(B_bsrColInd);
    cudaFree(B_bsrRowPtr);

    cudaFree(tA);
    cudaFree(A_bsrColInd);
    cudaFree(A_bsrRowPtr);


    // free cusparse csr spmv
//    cudaFree(C_csrVal);
//    cudaFree(C_csrColInd);
//    cudaFree(C_csrRowPtr);
//    cudaFree(B_csrVal);
//    cudaFree(B_csrColInd);
//    cudaFree(B_csrRowPtr);
//    cudaFree(d_A_nnz);
//    cudaFree(A_csrVal);
//    cudaFree(A_csrColInd);
//    cudaFree(A_csrRowPtr);
//    cusparseDestroyMatDescr(C_descr);
//    cusparseDestroyMatDescr(B_descr);
//    cusparseDestroyMatDescr(A_descr);
//    cusparseDestroy(handle_csr);

    // free mem
    cudaFree(csrVal);
    cudaFree(csrColInd);
    cudaFree(csrRowPtr);

    free(h_csrVal);
    free(h_csrColInd);
    free(h_csrRowPtr);

    // free all results
}

int main(int argc, char* argv[])
{
    main32(argc, argv);
    //main64(argc, argv);
}
