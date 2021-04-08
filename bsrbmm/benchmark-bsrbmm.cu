#include <iostream>
#include <sys/time.h>

#define TEST_TIMES 1
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#include "mmio_highlevel.h"
#include "csr2bsr_batch.cu"

//======================================================================================
// bsrbmm4
//======================================================================================
int main4(int argc, char* argv[])
{

    cudaSetDevice(0);
    if (argc < 2)
    {
        printf("./exe [xxx.mtx]\n");
        exit(1);
    }

    // ============================================= matrix storage
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
    // force all csrval to be 1 (this is for handling weighted adjacency matrix)
    setDeviceValArr<int, float><<<1,1>>>(csrVal, nnz, 1.0);
//    removeDiagonalNnz<<<1,1>>>(csrRowPtr, csrColInd, csrVal, nrows);

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
    unsigned csrbytes = (nrows+1+A_nnz*2) * 4 * 2;
    printf("csr total size: "); printBytes(csrbytes); printf("\n");

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
	int blocksize = 4;
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
    cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

    // csr2bsr in row-major order, estimate first
    int nblocks;

    cudaMalloc((void**)&A_bsrRowPtr, sizeof(int) *(nblockrows+1));
    cusparseXcsr2bsrNnz(handle, dirA, nrows, ncols, csr_descr,
                        A_csrRowPtr, A_csrColInd, blocksize, bsr_descr, A_bsrRowPtr, &nblocks);
    cudaMalloc((void**)&A_bsrColInd, sizeof(int)*nblocks);
    printf("blocksize: %d, nblockrows: %d, nblocks: %d\n", blocksize, nblockrows, nblocks);
    unsigned bytes = (nblocks * blocksize * 1 + (nblockrows+1+nblocks) * 4) * 2;
    printf("bsr total size: "); printBytes(bytes); printf("\n");

    // free cusparse descr and handle memory
    cusparseDestroyMatDescr(csr_descr);
    cusparseDestroyMatDescr(bsr_descr);
    cusparseDestroy(handle);

    // packed matrix tA
    uchar* tA;
    cudaMalloc((void**)&tA, nblocks * blocksize * sizeof(uchar));
    csr2bsr_batch_4_row(h_csrRowPtr, h_csrColInd, nrows, ncols, A_nnz,
                     A_bsrRowPtr, A_bsrColInd, tA, blocksize, nblockrows, nblocks); //row-major

    // csr2csc for B as A^T
    int* B_cscRowInd, *B_cscColPtr;
    float* B_cscVal;
    cudaMalloc(&B_cscRowInd, sizeof(int) * A_nnz);
    cudaMalloc(&B_cscColPtr, sizeof(int) * (nrows+1));
    cudaMalloc(&B_cscVal, sizeof(float) * A_nnz);

    cusparseHandle_t handle_csr2csc;
    cusparseCreate(&handle_csr2csc);
    cusparseScsr2csc(handle_csr2csc, nrows, ncols, A_nnz,
                     A_csrVal, A_csrRowPtr, A_csrColInd,
                     B_cscVal, B_cscRowInd, B_cscColPtr,
                     CUSPARSE_ACTION_NUMERIC,
                     CUSPARSE_INDEX_BASE_ZERO);
    cusparseDestroy(handle_csr2csc);

    int *h_B_cscRowInd, *h_B_cscColPtr;
    h_B_cscRowInd = (int*) malloc(sizeof(int) * A_nnz);
    h_B_cscColPtr = (int*) malloc(sizeof(int) * (nrows+1));
    cudaMemcpy(h_B_cscRowInd, B_cscRowInd, sizeof(int) * A_nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B_cscColPtr, B_cscColPtr, sizeof(int) * (nrows+1), cudaMemcpyDeviceToHost);

    // csr2bsr for B & pack matrix for tB
    int* B_bsrRowPtr, *B_bsrColInd;
    cudaMalloc(&B_bsrRowPtr, sizeof(int) * (nblockrows+1));
    cudaMalloc(&B_bsrColInd, sizeof(int) * nblocks);
    uchar* tB;
    cudaMalloc((void**)&tB, nblocks * blocksize * sizeof(uchar));
    csr2bsr_batch_4_col(h_B_cscColPtr, h_B_cscRowInd, nrows, ncols, A_nnz,
                     B_bsrRowPtr, B_bsrColInd, tB, blocksize, nblockrows, nblocks); //col-major
    free(h_B_cscRowInd);
    free(h_B_cscColPtr);


    // ============================================= BSTC-4 bsr bmm
    // allocate bsr storage for resulting C
    // use 1 float to store the reduced sum for now
    int* fC;
	cudaMalloc((void**)&fC, sizeof(int) * 1);
	setDeviceValArr<int, int><<<1,1>>>(fC, 1, 0);

    // get grid dim
    int gridDim = (int)ceil(cbrt((double)nblockrows));
    dim3 grid(gridDim, gridDim, gridDim);
#ifdef VERBOSE
    printf("cbrt(nblockrows) = %d\n", gridDim);
#endif

int *runtime;
#ifdef PROF
    cudaMalloc(&runtime, nblockrows * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(runtime, nblockrows, 0);
#endif

    // ------

    GpuTimer bmm_timer;
    bmm_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) { // follow warp consolidation model (32 threads per block)

//        bmm4_sparse<int, int><<<grid, 32>>>(tA, tB, fC,
//                                               A_bsrRowPtr, A_bsrColInd,
//                                               B_bsrRowPtr, B_bsrColInd,
//                                               nblockrows, nblocks, nrows);
        bmm4_sparse_masked_v4<int, int><<<grid, 32>>>(tA, tB, fC,
                                                   A_bsrRowPtr, A_bsrColInd,
                                                   B_bsrRowPtr, B_bsrColInd,
                                                   nblockrows, nblocks, nrows);
    }

    bmm_timer.Stop();
    double bmm4_time = bmm_timer.ElapsedMillis()/double(TEST_TIMES);

    // ------

    int ntris_bmm;
    cudaMemcpy(&ntris_bmm, fC, sizeof(int) * 1, cudaMemcpyDeviceToHost);

    printf("ntris_bmm: %d\n", ntris_bmm);
    printf("BSR BMM-4: %.3lf\n", bmm4_time);

    // ============================================= cuSPARSE csr spgemm-float
    // setup cusparse metadata
    cusparseHandle_t handle_csr;
    cusparseCreate(&handle_csr);

    cusparseMatDescr_t A_descr;
    cusparseCreateMatDescr(&A_descr);
    cusparseSetMatType(A_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(A_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t B_descr;
    cusparseCreateMatDescr(&B_descr);
    cusparseSetMatType(B_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(B_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t C_descr;
    cusparseCreateMatDescr(&C_descr);
    cusparseSetMatType(C_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(C_descr, CUSPARSE_INDEX_BASE_ZERO);

    // setup input and output csr storage
    // 2) get B = A^T (get B = A here, and transpose using CUSPARSE_OPERATION_TRANSPOSE)
    // duplicate A as B
    int* B_csrRowPtr, *B_csrColInd;
    float* B_csrVal;
    int B_nnz = A_nnz;
    cudaMalloc(&B_csrRowPtr, sizeof(int) * (nrows+1));
    cudaMalloc(&B_csrColInd, sizeof(int) * B_nnz);
    cudaMalloc(&B_csrVal, sizeof(float) * B_nnz);
    cudaMemcpy(B_csrRowPtr, B_cscColPtr, sizeof(int) * (nrows+1), cudaMemcpyDeviceToDevice);
    cudaMemcpy(B_csrColInd, B_cscRowInd, sizeof(int) * B_nnz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(B_csrVal, B_cscVal, sizeof(float) * B_nnz, cudaMemcpyDeviceToDevice);

    // calculate nnz in C and allocate storage
    int* C_csrRowPtr, *C_csrColInd;
    float* C_csrVal;
    int C_nnz;
    cudaMalloc(&C_csrRowPtr, sizeof(int) * (nrows+1));
    cusparseXcsrgemmNnz(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, nrows, nrows, nrows,
                        A_descr, A_nnz, A_csrRowPtr, A_csrColInd,
                        B_descr, B_nnz, B_csrRowPtr, B_csrColInd,
                        C_descr, C_csrRowPtr, &C_nnz);

    cudaMalloc(&C_csrColInd, sizeof(int) * C_nnz);
    setDeviceValArr<int, int><<<1,1>>>(C_csrColInd, C_nnz, 0);
    cudaMalloc(&C_csrVal, sizeof(float) * C_nnz);
    setDeviceValArr<int, float><<<1,1>>>(C_csrVal, C_nnz, 0);
//    printf("result C_csrVal nnz: %d\n", C_nnz);

    // ------

    GpuTimer csr_timer;
    csr_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) {

//       bmm4_sparse_full<int, float><<<grid, 32>>>(tA, tB, C_csrVal,
//                                            A_bsrRowPtr, A_bsrColInd,
//                                            B_bsrRowPtr, B_bsrColInd,
//                                            C_csrRowPtr, C_csrColInd,
//                                            nblockrows, nblocks, nrows);

        cusparseScsrgemm(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, nrows, nrows, nrows,
                         A_descr, A_nnz, A_csrVal, A_csrRowPtr, A_csrColInd,
                         B_descr, B_nnz, B_csrVal, B_csrRowPtr, B_csrColInd,
                         C_descr, C_csrVal, C_csrRowPtr, C_csrColInd);
    }

    csr_timer.Stop();
    double cusparsecsrspgemmfloat_time = csr_timer.ElapsedMillis()/double(TEST_TIMES);

//    printResVec<<<1,1>>>(C_csrVal, C_nnz);

    // ------

    int* result_cusparsecsrspgemmfloat;
    cudaMalloc((void**)&result_cusparsecsrspgemmfloat, sizeof(int) * 1);
    reuduceSum<float><<<1,1>>>(C_csrVal, C_nnz, result_cusparsecsrspgemmfloat);
    int ntris_csr;
    cudaMemcpy(&ntris_csr, result_cusparsecsrspgemmfloat, sizeof(int) * 1, cudaMemcpyDeviceToHost);
    printf("ntris_csr: %d\n", ntris_csr);


    //============================================= check result
    //printf("CuSPARSE CSR SpGEMM-float (baseline) success: %d\n", check_result(result_cusparsebsrspmvfloat, result_cusparsebsrspmvfloat, ncols));
    //printf("BSR BMM-4 success: %d\n", check_result(result_bsrbmv4, result_cusparsebsrspmvfloat, ncols));

    printf("CuSPARSE CSR SpGEMM-float: %.3lf\n", cusparsecsrspgemmfloat_time);

    //============================================= free memory
    // free bsr bmm
    cudaFree(fC);

    cudaFree(tB);
    cudaFree(B_bsrColInd);
    cudaFree(B_bsrRowPtr);

    cudaFree(tA);
    cudaFree(A_bsrColInd);
    cudaFree(A_bsrRowPtr);


    // free cusparse csr spmv
    cudaFree(C_csrVal);
    cudaFree(C_csrColInd);
    cudaFree(C_csrRowPtr);
    cudaFree(B_csrVal);
    cudaFree(B_csrColInd);
    cudaFree(B_csrRowPtr);
    cudaFree(d_A_nnz);
    cudaFree(A_csrVal);
    cudaFree(A_csrColInd);
    cudaFree(A_csrRowPtr);
    cusparseDestroyMatDescr(C_descr);
    cusparseDestroyMatDescr(B_descr);
    cusparseDestroyMatDescr(A_descr);
    cusparseDestroy(handle_csr);

    // free mem
    cudaFree(csrVal);
    cudaFree(csrColInd);
    cudaFree(csrRowPtr);

    free(h_csrVal);
    free(h_csrColInd);
    free(h_csrRowPtr);

    // free all results
}


//======================================================================================
// bsrbmm8
//======================================================================================
int main8(int argc, char* argv[])
{

    cudaSetDevice(0);
    if (argc < 2)
    {
        printf("./exe [xxx.mtx]\n");
        exit(1);
    }

    // ============================================= matrix storage
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
    // force all csrval to be 1 (this is for handling weighted adjacency matrix)
    setDeviceValArr<int, float><<<1,1>>>(csrVal, nnz, 1.0);
//    removeDiagonalNnz<<<1,1>>>(csrRowPtr, csrColInd, csrVal, nrows);

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
    unsigned csrbytes = (nrows+1+A_nnz*2) * 4 * 2;
    printf("csr total size: "); printBytes(csrbytes); printf("\n");

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
	int blocksize = 8;
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
    cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

    // csr2bsr in row-major order, estimate first
    int nblocks;

    cudaMalloc((void**)&A_bsrRowPtr, sizeof(int) *(nblockrows+1));
    cusparseXcsr2bsrNnz(handle, dirA, nrows, ncols, csr_descr,
                        A_csrRowPtr, A_csrColInd, blocksize, bsr_descr, A_bsrRowPtr, &nblocks);
    cudaMalloc((void**)&A_bsrColInd, sizeof(int)*nblocks);
    printf("blocksize: %d, nblockrows: %d, nblocks: %d\n", blocksize, nblockrows, nblocks);
    unsigned bytes = (nblocks * blocksize * 1 + (nblockrows+1+nblocks) * 4) * 2;
    printf("bsr total size: "); printBytes(bytes); printf("\n");

    // free cusparse descr and handle memory
    cusparseDestroyMatDescr(csr_descr);
    cusparseDestroyMatDescr(bsr_descr);
    cusparseDestroy(handle);

    // packed matrix tA
    uchar* tA;
    cudaMalloc((void**)&tA, nblocks * blocksize * sizeof(uchar));
    csr2bsr_batch_8_row(h_csrRowPtr, h_csrColInd, nrows, ncols, A_nnz,
                     A_bsrRowPtr, A_bsrColInd, tA, blocksize, nblockrows, nblocks); //row-major

    // csr2csc for B as A^T
    int* B_cscRowInd, *B_cscColPtr;
    float* B_cscVal;
    cudaMalloc(&B_cscRowInd, sizeof(int) * A_nnz);
    cudaMalloc(&B_cscColPtr, sizeof(int) * (nrows+1));
    cudaMalloc(&B_cscVal, sizeof(float) * A_nnz);

    cusparseHandle_t handle_csr2csc;
    cusparseCreate(&handle_csr2csc);
    cusparseScsr2csc(handle_csr2csc, nrows, ncols, A_nnz,
                     A_csrVal, A_csrRowPtr, A_csrColInd,
                     B_cscVal, B_cscRowInd, B_cscColPtr,
                     CUSPARSE_ACTION_NUMERIC,
                     CUSPARSE_INDEX_BASE_ZERO);
    cusparseDestroy(handle_csr2csc);

    int *h_B_cscRowInd, *h_B_cscColPtr;
    h_B_cscRowInd = (int*) malloc(sizeof(int) * A_nnz);
    h_B_cscColPtr = (int*) malloc(sizeof(int) * (nrows+1));
    cudaMemcpy(h_B_cscRowInd, B_cscRowInd, sizeof(int) * A_nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B_cscColPtr, B_cscColPtr, sizeof(int) * (nrows+1), cudaMemcpyDeviceToHost);

    // csr2bsr for B & pack matrix for tB
    int* B_bsrRowPtr, *B_bsrColInd;
    cudaMalloc(&B_bsrRowPtr, sizeof(int) * (nblockrows+1));
    cudaMalloc(&B_bsrColInd, sizeof(int) * nblocks);
    uchar* tB;
    cudaMalloc((void**)&tB, nblocks * blocksize * sizeof(uchar));
    csr2bsr_batch_8_col(h_B_cscColPtr, h_B_cscRowInd, nrows, ncols, A_nnz,
                     B_bsrRowPtr, B_bsrColInd, tB, blocksize, nblockrows, nblocks); //col-major
    free(h_B_cscRowInd);
    free(h_B_cscColPtr);


    // ============================================= BSTC-8 bsr bmm
    // allocate bsr storage for resulting C
    // use 1 float to store the reduced sum for now
    int* fC;
	cudaMalloc((void**)&fC, sizeof(int) * 1);
	setDeviceValArr<int, int><<<1,1>>>(fC, 1, 0);

    // get grid dim
    int gridDim = (int)ceil(cbrt((double)nblockrows));
    dim3 grid(gridDim, gridDim, gridDim);
#ifdef VERBOSE
    printf("cbrt(nblockrows) = %d\n", gridDim);
#endif

int *runtime;
#ifdef PROF
    cudaMalloc(&runtime, nblockrows * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(runtime, nblockrows, 0);
#endif

    // ------

    GpuTimer bmm_timer;
    bmm_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) { // follow warp consolidation model (32 threads per block)

        bmm8_sparse<int, int><<<grid, 32>>>(tA, tB, fC,
                                               A_bsrRowPtr, A_bsrColInd,
                                               B_bsrRowPtr, B_bsrColInd,
                                               nblockrows, nblocks, nrows, runtime);
    }

    bmm_timer.Stop();
    double bmm8_time = bmm_timer.ElapsedMillis()/double(TEST_TIMES);

    // ------
#ifdef PROF
    printTimeReport<<<1,1>>>(runtime, nblockrows); cudaFree(runtime);
#endif

    int ntris_bmm;
    cudaMemcpy(&ntris_bmm, fC, sizeof(int) * 1, cudaMemcpyDeviceToHost);

    printf("ntris_bmm: %d\n", ntris_bmm);
    printf("BSR BMM-8: %.3lf\n", bmm8_time);

    // ============================================= cuSPARSE csr spgemm-float
    // setup cusparse metadata
    cusparseHandle_t handle_csr;
    cusparseCreate(&handle_csr);

    cusparseMatDescr_t A_descr;
    cusparseCreateMatDescr(&A_descr);
    cusparseSetMatType(A_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(A_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t B_descr;
    cusparseCreateMatDescr(&B_descr);
    cusparseSetMatType(B_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(B_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t C_descr;
    cusparseCreateMatDescr(&C_descr);
    cusparseSetMatType(C_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(C_descr, CUSPARSE_INDEX_BASE_ZERO);

    // setup input and output csr storage
    // 2) get B = A^T (get B = A here, and transpose using CUSPARSE_OPERATION_TRANSPOSE)
    // duplicate A as B
    int* B_csrRowPtr, *B_csrColInd;
    float* B_csrVal;
    int B_nnz = A_nnz;
    cudaMalloc(&B_csrRowPtr, sizeof(int) * (nrows+1));
    cudaMalloc(&B_csrColInd, sizeof(int) * B_nnz);
    cudaMalloc(&B_csrVal, sizeof(float) * B_nnz);
    cudaMemcpy(B_csrRowPtr, B_cscColPtr, sizeof(int) * (nrows+1), cudaMemcpyDeviceToDevice);
    cudaMemcpy(B_csrColInd, B_cscRowInd, sizeof(int) * B_nnz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(B_csrVal, B_cscVal, sizeof(float) * B_nnz, cudaMemcpyDeviceToDevice);

    // calculate nnz in C and allocate storage
    int* C_csrRowPtr, *C_csrColInd;
    float* C_csrVal;
    int C_nnz;
    cudaMalloc(&C_csrRowPtr, sizeof(int) * (nrows+1));
    cusparseXcsrgemmNnz(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, nrows, nrows, nrows,
                        A_descr, A_nnz, A_csrRowPtr, A_csrColInd,
                        B_descr, B_nnz, B_csrRowPtr, B_csrColInd,
                        C_descr, C_csrRowPtr, &C_nnz);

    cudaMalloc(&C_csrColInd, sizeof(int) * C_nnz);
    cudaMalloc(&C_csrVal, sizeof(float) * C_nnz);

    // ------

    GpuTimer csr_timer;
    csr_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) {
        cusparseScsrgemm(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, nrows, nrows, nrows,
                         A_descr, A_nnz, A_csrVal, A_csrRowPtr, A_csrColInd,
                         B_descr, B_nnz, B_csrVal, B_csrRowPtr, B_csrColInd,
                         C_descr, C_csrVal, C_csrRowPtr, C_csrColInd);
    }

    csr_timer.Stop();
    double cusparsecsrspgemmfloat_time = csr_timer.ElapsedMillis()/double(TEST_TIMES);

    // ------

    int* result_cusparsecsrspgemmfloat;
    cudaMalloc((void**)&result_cusparsecsrspgemmfloat, sizeof(int) * 1);
    reuduceSum<float><<<1,1>>>(C_csrVal, C_nnz, result_cusparsecsrspgemmfloat);
    int ntris_csr;
    cudaMemcpy(&ntris_csr, result_cusparsecsrspgemmfloat, sizeof(int) * 1, cudaMemcpyDeviceToHost);
    printf("ntris_csr: %d\n", ntris_csr);


    //============================================= check result
    //printf("CuSPARSE CSR SpGEMM-float (baseline) success: %d\n", check_result(result_cusparsebsrspmvfloat, result_cusparsebsrspmvfloat, ncols));
    //printf("BSR BMM-8 success: %d\n", check_result(result_bsrbmv8, result_cusparsebsrspmvfloat, ncols));

    printf("CuSPARSE CSR SpGEMM-float: %.3lf\n", cusparsecsrspgemmfloat_time);

    //============================================= free memory
    // free bsr bmm
    cudaFree(fC);

    cudaFree(tB);
    cudaFree(B_bsrColInd);
    cudaFree(B_bsrRowPtr);

    cudaFree(tA);
    cudaFree(A_bsrColInd);
    cudaFree(A_bsrRowPtr);


    // free cusparse csr spmv
    cudaFree(C_csrVal);
    cudaFree(C_csrColInd);
    cudaFree(C_csrRowPtr);
    cudaFree(B_csrVal);
    cudaFree(B_csrColInd);
    cudaFree(B_csrRowPtr);
    cudaFree(d_A_nnz);
    cudaFree(A_csrVal);
    cudaFree(A_csrColInd);
    cudaFree(A_csrRowPtr);
    cusparseDestroyMatDescr(C_descr);
    cusparseDestroyMatDescr(B_descr);
    cusparseDestroyMatDescr(A_descr);
    cusparseDestroy(handle_csr);

    // free mem
    cudaFree(csrVal);
    cudaFree(csrColInd);
    cudaFree(csrRowPtr);

    free(h_csrVal);
    free(h_csrColInd);
    free(h_csrRowPtr);

    // free all results
}

//======================================================================================
// bsrbmm16
//======================================================================================
int main16(int argc, char* argv[])
{

    cudaSetDevice(0);
    if (argc < 2)
    {
        printf("./exe [xxx.mtx]\n");
        exit(1);
    }

    // ============================================= matrix storage
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
    // force all csrval to be 1 (this is for handling weighted adjacency matrix)
    setDeviceValArr<int, float><<<1,1>>>(csrVal, nnz, 1.0);
//    removeDiagonalNnz<<<1,1>>>(csrRowPtr, csrColInd, csrVal, nrows);

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
    unsigned csrbytes = (nrows+1+A_nnz*2) * 4 * 2;
    printf("csr total size: "); printBytes(csrbytes); printf("\n");

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
	int blocksize = 16;
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
    cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

    // csr2bsr in row-major order, estimate first
    int nblocks;

    cudaMalloc((void**)&A_bsrRowPtr, sizeof(int) *(nblockrows+1));
    cusparseXcsr2bsrNnz(handle, dirA, nrows, ncols, csr_descr,
                        A_csrRowPtr, A_csrColInd, blocksize, bsr_descr, A_bsrRowPtr, &nblocks);
    cudaMalloc((void**)&A_bsrColInd, sizeof(int)*nblocks);
    printf("blocksize: %d, nblockrows: %d, nblocks: %d\n", blocksize, nblockrows, nblocks);
    unsigned bytes = (nblocks * blocksize * 2 + (nblockrows+1+nblocks) * 4) * 2;
    printf("bsr total size: "); printBytes(bytes); printf("\n");

    // free cusparse descr and handle memory
    cusparseDestroyMatDescr(csr_descr);
    cusparseDestroyMatDescr(bsr_descr);
    cusparseDestroy(handle);

    // packed matrix tA
    ushort* tA;
    cudaMalloc((void**)&tA, nblocks * blocksize * sizeof(ushort));
    csr2bsr_batch_16_row(h_csrRowPtr, h_csrColInd, nrows, ncols, A_nnz,
                     A_bsrRowPtr, A_bsrColInd, tA, blocksize, nblockrows, nblocks); //row-major

    // csr2csc for B as A^T
    int* B_cscRowInd, *B_cscColPtr;
    float* B_cscVal;
    cudaMalloc(&B_cscRowInd, sizeof(int) * A_nnz);
    cudaMalloc(&B_cscColPtr, sizeof(int) * (nrows+1));
    cudaMalloc(&B_cscVal, sizeof(float) * A_nnz);

    cusparseHandle_t handle_csr2csc;
    cusparseCreate(&handle_csr2csc);
    cusparseScsr2csc(handle_csr2csc, nrows, ncols, A_nnz,
                     A_csrVal, A_csrRowPtr, A_csrColInd,
                     B_cscVal, B_cscRowInd, B_cscColPtr,
                     CUSPARSE_ACTION_NUMERIC,
                     CUSPARSE_INDEX_BASE_ZERO);
    cusparseDestroy(handle_csr2csc);

    int *h_B_cscRowInd, *h_B_cscColPtr;
    h_B_cscRowInd = (int*) malloc(sizeof(int) * A_nnz);
    h_B_cscColPtr = (int*) malloc(sizeof(int) * (nrows+1));
    cudaMemcpy(h_B_cscRowInd, B_cscRowInd, sizeof(int) * A_nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B_cscColPtr, B_cscColPtr, sizeof(int) * (nrows+1), cudaMemcpyDeviceToHost);

    // csr2bsr for B & pack matrix for tB
    int* B_bsrRowPtr, *B_bsrColInd;
    cudaMalloc(&B_bsrRowPtr, sizeof(int) * (nblockrows+1));
    cudaMalloc(&B_bsrColInd, sizeof(int) * nblocks);
    ushort* tB;
    cudaMalloc((void**)&tB, nblocks * blocksize * sizeof(ushort));
    csr2bsr_batch_16_col(h_B_cscColPtr, h_B_cscRowInd, nrows, ncols, A_nnz,
                     B_bsrRowPtr, B_bsrColInd, tB, blocksize, nblockrows, nblocks); //col-major
    free(h_B_cscRowInd);
    free(h_B_cscColPtr);

    // ============================================= BSTC-16 bsr bmm
    // allocate bsr storage for resulting C
    // use 1 float to store the reduced sum for now
    int* fC;
	cudaMalloc((void**)&fC, sizeof(int) * 1);
	setDeviceValArr<int, int><<<1,1>>>(fC, 1, 0);

    // get grid dim
    int gridDim = (int)ceil(cbrt((double)nblockrows));
    dim3 grid(gridDim, gridDim, gridDim);
#ifdef VERBOSE
    printf("cbrt(nblockrows) = %d\n", gridDim);
#endif

int *runtime;
#ifdef PROF
    cudaMalloc(&runtime, nblockrows * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(runtime, nblockrows, 0);
#endif

    // ------

    GpuTimer bmm_timer;
    bmm_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) { // follow warp consolidation model (32 threads per block)

        bmm16_sparse<int, int><<<grid, 32>>>(tA, tB, fC,
                                               A_bsrRowPtr, A_bsrColInd,
                                               B_bsrRowPtr, B_bsrColInd,
                                               nblockrows, nblocks, nrows, runtime);
    }

    bmm_timer.Stop();
    double bmm16_time = bmm_timer.ElapsedMillis()/double(TEST_TIMES);

    // ------
#ifdef PROF
    printTimeReport<<<1,1>>>(runtime, nblockrows); cudaFree(runtime);
#endif


    int ntris_bmm;
    cudaMemcpy(&ntris_bmm, fC, sizeof(int) * 1, cudaMemcpyDeviceToHost);

    printf("ntris_bmm: %d\n", ntris_bmm);
    printf("BSR BMM-16: %.3lf\n", bmm16_time);

    // ============================================= cuSPARSE csr spgemm-float
    // setup cusparse metadata
    cusparseHandle_t handle_csr;
    cusparseCreate(&handle_csr);

    cusparseMatDescr_t A_descr;
    cusparseCreateMatDescr(&A_descr);
    cusparseSetMatType(A_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(A_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t B_descr;
    cusparseCreateMatDescr(&B_descr);
    cusparseSetMatType(B_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(B_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t C_descr;
    cusparseCreateMatDescr(&C_descr);
    cusparseSetMatType(C_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(C_descr, CUSPARSE_INDEX_BASE_ZERO);

    // setup input and output csr storage
    // 2) get B = A^T (get B = A here, and transpose using CUSPARSE_OPERATION_TRANSPOSE)
    // duplicate A as B
    int* B_csrRowPtr, *B_csrColInd;
    float* B_csrVal;
    int B_nnz = A_nnz;
    cudaMalloc(&B_csrRowPtr, sizeof(int) * (nrows+1));
    cudaMalloc(&B_csrColInd, sizeof(int) * B_nnz);
    cudaMalloc(&B_csrVal, sizeof(float) * B_nnz);
    cudaMemcpy(B_csrRowPtr, B_cscColPtr, sizeof(int) * (nrows+1), cudaMemcpyDeviceToDevice);
    cudaMemcpy(B_csrColInd, B_cscRowInd, sizeof(int) * B_nnz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(B_csrVal, B_cscVal, sizeof(float) * B_nnz, cudaMemcpyDeviceToDevice);

    // calculate nnz in C and allocate storage
    int* C_csrRowPtr, *C_csrColInd;
    float* C_csrVal;
    int C_nnz;
    cudaMalloc(&C_csrRowPtr, sizeof(int) * (nrows+1));
    cusparseXcsrgemmNnz(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, nrows, nrows, nrows,
                        A_descr, A_nnz, A_csrRowPtr, A_csrColInd,
                        B_descr, B_nnz, B_csrRowPtr, B_csrColInd,
                        C_descr, C_csrRowPtr, &C_nnz);

    cudaMalloc(&C_csrColInd, sizeof(int) * C_nnz);
    cudaMalloc(&C_csrVal, sizeof(float) * C_nnz);

    // ------

    GpuTimer csr_timer;
    csr_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) {
        cusparseScsrgemm(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, nrows, nrows, nrows,
                         A_descr, A_nnz, A_csrVal, A_csrRowPtr, A_csrColInd,
                         B_descr, B_nnz, B_csrVal, B_csrRowPtr, B_csrColInd,
                         C_descr, C_csrVal, C_csrRowPtr, C_csrColInd);
    }

    csr_timer.Stop();
    double cusparsecsrspgemmfloat_time = csr_timer.ElapsedMillis()/double(TEST_TIMES);

    // ------

    int* result_cusparsecsrspgemmfloat;
    cudaMalloc((void**)&result_cusparsecsrspgemmfloat, sizeof(int) * 1);
    reuduceSum<float><<<1,1>>>(C_csrVal, C_nnz, result_cusparsecsrspgemmfloat);
    int ntris_csr;
    cudaMemcpy(&ntris_csr, result_cusparsecsrspgemmfloat, sizeof(int) * 1, cudaMemcpyDeviceToHost);
    printf("ntris_csr: %d\n", ntris_csr);


    //============================================= check result
    //printf("CuSPARSE CSR SpGEMM-float (baseline) success: %d\n", check_result(result_cusparsebsrspmvfloat, result_cusparsebsrspmvfloat, ncols));
    //printf("BSR BMM-16 success: %d\n", check_result(result_bsrbmv16, result_cusparsebsrspmvfloat, ncols));

    printf("CuSPARSE CSR SpGEMM-float: %.3lf\n", cusparsecsrspgemmfloat_time);

    //============================================= free memory
    // free bsr bmm
    cudaFree(fC);

    cudaFree(tB);
    cudaFree(B_bsrColInd);
    cudaFree(B_bsrRowPtr);

    cudaFree(tA);
    cudaFree(A_bsrColInd);
    cudaFree(A_bsrRowPtr);


    // free cusparse csr spmv
    cudaFree(C_csrVal);
    cudaFree(C_csrColInd);
    cudaFree(C_csrRowPtr);
    cudaFree(B_csrVal);
    cudaFree(B_csrColInd);
    cudaFree(B_csrRowPtr);
    cudaFree(d_A_nnz);
    cudaFree(A_csrVal);
    cudaFree(A_csrColInd);
    cudaFree(A_csrRowPtr);
    cusparseDestroyMatDescr(C_descr);
    cusparseDestroyMatDescr(B_descr);
    cusparseDestroyMatDescr(A_descr);
    cusparseDestroy(handle_csr);

    // free mem
    cudaFree(csrVal);
    cudaFree(csrColInd);
    cudaFree(csrRowPtr);

    free(h_csrVal);
    free(h_csrColInd);
    free(h_csrRowPtr);

    // free all results
}

//======================================================================================
// bsrbmm32
//======================================================================================
int main32(int argc, char* argv[])
{

    cudaSetDevice(0);
    if (argc < 2)
    {
        printf("./exe [xxx.mtx]\n");
        exit(1);
    }

    // ============================================= matrix storage
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
    // force all csrval to be 1 (this is for handling weighted adjacency matrix)
    setDeviceValArr<int, float><<<1,1>>>(csrVal, nnz, 1.0);
//    removeDiagonalNnz<<<1,1>>>(csrRowPtr, csrColInd, csrVal, nrows);

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
    unsigned csrbytes = (nrows+1+A_nnz*2) * 4 * 2;
    printf("csr total size: "); printBytes(csrbytes); printf("\n");

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
    cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

    // csr2bsr in row-major order, estimate first
    int nblocks;

    cudaMalloc((void**)&A_bsrRowPtr, sizeof(int) *(nblockrows+1));
    cusparseXcsr2bsrNnz(handle, dirA, nrows, ncols, csr_descr,
                        A_csrRowPtr, A_csrColInd, blocksize, bsr_descr, A_bsrRowPtr, &nblocks);
    cudaMalloc((void**)&A_bsrColInd, sizeof(int)*nblocks);
    printf("blocksize: %d, nblockrows: %d, nblocks: %d\n", blocksize, nblockrows, nblocks);
    unsigned bytes = (nblocks * blocksize * 4 + (nblockrows+1+nblocks) * 4) * 2;
    printf("bsr total size: "); printBytes(bytes); printf("\n");

    // free cusparse descr and handle memory
    cusparseDestroyMatDescr(csr_descr);
    cusparseDestroyMatDescr(bsr_descr);
    cusparseDestroy(handle);

    // packed matrix tA
    unsigned* tA;
    cudaMalloc((void**)&tA, nblocks * blocksize * sizeof(unsigned));
    csr2bsr_batch_32_row(h_csrRowPtr, h_csrColInd, nrows, ncols, A_nnz,
                     A_bsrRowPtr, A_bsrColInd, tA, blocksize, nblockrows, nblocks); //row-major

    // csr2csc for B as A^T
    int* B_cscRowInd, *B_cscColPtr;
    float* B_cscVal;
    cudaMalloc(&B_cscRowInd, sizeof(int) * A_nnz);
    cudaMalloc(&B_cscColPtr, sizeof(int) * (nrows+1));
    cudaMalloc(&B_cscVal, sizeof(float) * A_nnz);

    cusparseHandle_t handle_csr2csc;
    cusparseCreate(&handle_csr2csc);
    cusparseScsr2csc(handle_csr2csc, nrows, ncols, A_nnz,
                     A_csrVal, A_csrRowPtr, A_csrColInd,
                     B_cscVal, B_cscRowInd, B_cscColPtr,
                     CUSPARSE_ACTION_NUMERIC,
                     CUSPARSE_INDEX_BASE_ZERO);
    cusparseDestroy(handle_csr2csc);

    int *h_B_cscRowInd, *h_B_cscColPtr;
    h_B_cscRowInd = (int*) malloc(sizeof(int) * A_nnz);
    h_B_cscColPtr = (int*) malloc(sizeof(int) * (nrows+1));
    cudaMemcpy(h_B_cscRowInd, B_cscRowInd, sizeof(int) * A_nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B_cscColPtr, B_cscColPtr, sizeof(int) * (nrows+1), cudaMemcpyDeviceToHost);

    // csr2bsr for B & pack matrix for tB
    int* B_bsrRowPtr, *B_bsrColInd;
    cudaMalloc(&B_bsrRowPtr, sizeof(int) * (nblockrows+1));
    cudaMalloc(&B_bsrColInd, sizeof(int) * nblocks);
    unsigned* tB;
    cudaMalloc((void**)&tB, nblocks * blocksize * sizeof(unsigned));
    csr2bsr_batch_32_col(h_B_cscColPtr, h_B_cscRowInd, nrows, ncols, A_nnz,
                     B_bsrRowPtr, B_bsrColInd, tB, blocksize, nblockrows, nblocks); //col-major
    free(h_B_cscRowInd);
    free(h_B_cscColPtr);

    // ============================================= BSTC-32 bsr bmm
    // allocate bsr storage for resulting C
    // use 1 float to store the reduced sum for now
    int* fC;
	cudaMalloc((void**)&fC, sizeof(int) * 1);
	setDeviceValArr<int, int><<<1,1>>>(fC, 1, 0);

    // get grid dim
    int gridDim = (int)ceil(cbrt((double)nblockrows));
    dim3 grid(gridDim, gridDim, gridDim);

#ifdef VERBOSE
    printf("cbrt(nblockrows) = %d\n", gridDim);
#endif

int *runtime;
#ifdef PROF
    cudaMalloc(&runtime, nblockrows * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(runtime, nblockrows, 0);
#endif

    // ------

    GpuTimer bmm_timer;
    bmm_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) { // follow warp consolidation model (32 threads per block)

//        bmm32_sparse<int, int><<<grid, 32>>>(tA, tB, fC,
//                                           A_bsrRowPtr, A_bsrColInd,
//                                           B_bsrRowPtr, B_bsrColInd,
//                                           nblockrows, nblocks, nrows, runtime);

//        bmm32_sparse_masked<int, int><<<grid, 32>>>(tA, tB, fC,
//                                                   A_bsrRowPtr, A_bsrColInd,
//                                                   B_bsrRowPtr, B_bsrColInd,
//                                                   nblockrows, nblocks, nrows);

//        bmm32_sparse_masked_v3<int, int><<<grid, 32>>>(tA, tB, fC,
//                                                    A_bsrRowPtr, A_bsrColInd,
//                                                    B_bsrRowPtr, B_bsrColInd,
//                                                    nblockrows, nblocks, nrows);

        bmm32_sparse_masked_v4<int, int><<<grid, 32>>>(tA, tB, fC,
                                                    A_bsrRowPtr, A_bsrColInd,
                                                    B_bsrRowPtr, B_bsrColInd,
                                                    nblockrows, nblocks, nrows);
    }

    bmm_timer.Stop();
    double bmm32_time = bmm_timer.ElapsedMillis()/double(TEST_TIMES);

    // ------
#ifdef PROF
    printTimeReport<<<1,1>>>(runtime, nblockrows); cudaFree(runtime);
#endif

    int ntris_bmm;
    cudaMemcpy(&ntris_bmm, fC, sizeof(int) * 1, cudaMemcpyDeviceToHost);

    printf("ntris_bmm: %d\n", ntris_bmm);
    printf("BSR BMM-32: %.3lf\n", bmm32_time);

    // ============================================= cuSPARSE csr spgemm-float
    // setup cusparse metadata
    cusparseHandle_t handle_csr;
    cusparseCreate(&handle_csr);

    cusparseMatDescr_t A_descr;
    cusparseCreateMatDescr(&A_descr);
    cusparseSetMatType(A_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(A_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t B_descr;
    cusparseCreateMatDescr(&B_descr);
    cusparseSetMatType(B_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(B_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t C_descr;
    cusparseCreateMatDescr(&C_descr);
    cusparseSetMatType(C_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(C_descr, CUSPARSE_INDEX_BASE_ZERO);

    // setup input and output csr storage
    // 2) get B = A^T (get B = A here, and transpose using CUSPARSE_OPERATION_TRANSPOSE)
    // duplicate A as B
    int* B_csrRowPtr, *B_csrColInd;
    float* B_csrVal;
    int B_nnz = A_nnz;
    cudaMalloc(&B_csrRowPtr, sizeof(int) * (nrows+1));
    cudaMalloc(&B_csrColInd, sizeof(int) * B_nnz);
    cudaMalloc(&B_csrVal, sizeof(float) * B_nnz);
    cudaMemcpy(B_csrRowPtr, B_cscColPtr, sizeof(int) * (nrows+1), cudaMemcpyDeviceToDevice);
    cudaMemcpy(B_csrColInd, B_cscRowInd, sizeof(int) * B_nnz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(B_csrVal, B_cscVal, sizeof(float) * B_nnz, cudaMemcpyDeviceToDevice);

    // calculate nnz in C and allocate storage
    int* C_csrRowPtr, *C_csrColInd;
    float* C_csrVal;
    int C_nnz;
    cudaMalloc(&C_csrRowPtr, sizeof(int) * (nrows+1));
    cusparseXcsrgemmNnz(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, nrows, nrows, nrows,
                        A_descr, A_nnz, A_csrRowPtr, A_csrColInd,
                        B_descr, B_nnz, B_csrRowPtr, B_csrColInd,
                        C_descr, C_csrRowPtr, &C_nnz);

    cudaMalloc(&C_csrColInd, sizeof(int) * C_nnz);
    setDeviceValArr<int, int><<<1,1>>>(C_csrColInd, C_nnz, 0);
    cudaMalloc(&C_csrVal, sizeof(float) * C_nnz);
    setDeviceValArr<int, float><<<1,1>>>(C_csrVal, C_nnz, 0);
//    printf("result C_csrVal nnz: %d\n", C_nnz);

    // ------

    GpuTimer csr_timer;
    csr_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) {


//       bmm32_sparse_full_v1<int, float><<<grid, 32>>>(tA, tB, C_csrVal,
//                                                A_bsrRowPtr, A_bsrColInd,
//                                                B_bsrRowPtr, B_bsrColInd,
//                                                C_csrRowPtr, C_csrColInd,
//                                                nblockrows, nblocks, nrows);

        cusparseScsrgemm(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, nrows, nrows, nrows,
                         A_descr, A_nnz, A_csrVal, A_csrRowPtr, A_csrColInd,
                         B_descr, B_nnz, B_csrVal, B_csrRowPtr, B_csrColInd,
                         C_descr, C_csrVal, C_csrRowPtr, C_csrColInd);
    }

    csr_timer.Stop();
    double cusparsecsrspgemmfloat_time = csr_timer.ElapsedMillis()/double(TEST_TIMES);

//    printResVec<<<1,1>>>(C_csrVal, C_nnz);
//    printResVec<<<1,1>>>(A_csrVal, A_nnz);

    // ------

    int* result_cusparsecsrspgemmfloat;
    cudaMalloc((void**)&result_cusparsecsrspgemmfloat, sizeof(int) * 1);
    reuduceSum<float><<<1,1>>>(C_csrVal, C_nnz, result_cusparsecsrspgemmfloat);
    int ntris_csr;
    cudaMemcpy(&ntris_csr, result_cusparsecsrspgemmfloat, sizeof(int) * 1, cudaMemcpyDeviceToHost);
    printf("ntris_csr: %d\n", ntris_csr);


    //============================================= check result
    //printf("CuSPARSE CSR SpGEMM-float (baseline) success: %d\n", check_result(result_cusparsebsrspmvfloat, result_cusparsebsrspmvfloat, ncols));
    //printf("BSR BMM-32 success: %d\n", check_result(result_bsrbmv32, result_cusparsebsrspmvfloat, ncols));

    printf("CuSPARSE CSR SpGEMM-float: %.3lf\n", cusparsecsrspgemmfloat_time);

    //============================================= free memory
    // free bsr bmm
    cudaFree(fC);

    cudaFree(tB);
    cudaFree(B_bsrColInd);
    cudaFree(B_bsrRowPtr);

    cudaFree(tA);
    cudaFree(A_bsrColInd);
    cudaFree(A_bsrRowPtr);


    // free cusparse csr spmv
    cudaFree(C_csrVal);
    cudaFree(C_csrColInd);
    cudaFree(C_csrRowPtr);
    cudaFree(B_csrVal);
    cudaFree(B_csrColInd);
    cudaFree(B_csrRowPtr);
    cudaFree(d_A_nnz);
    cudaFree(A_csrVal);
    cudaFree(A_csrColInd);
    cudaFree(A_csrRowPtr);
    cusparseDestroyMatDescr(C_descr);
    cusparseDestroyMatDescr(B_descr);
    cusparseDestroyMatDescr(A_descr);
    cusparseDestroy(handle_csr);

    // free mem
    cudaFree(csrVal);
    cudaFree(csrColInd);
    cudaFree(csrRowPtr);

    free(h_csrVal);
    free(h_csrColInd);
    free(h_csrRowPtr);

    // free all results
}

//======================================================================================
// bsrbmm64
//======================================================================================
int main64(int argc, char* argv[])
{

    cudaSetDevice(0);
    if (argc < 2)
    {
        printf("./exe [xxx.mtx]\n");
        exit(1);
    }

    // ============================================= matrix storage
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
    // force all csrval to be 1 (this is for handling weighted adjacency matrix)
    setDeviceValArr<int, float><<<1,1>>>(csrVal, nnz, 1.0);
//    removeDiagonalNnz<<<1,1>>>(csrRowPtr, csrColInd, csrVal, nrows);

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
    unsigned csrbytes = (nrows+1+A_nnz*2) * 4 * 2;
    printf("csr total size: "); printBytes(csrbytes); printf("\n");

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
	int blocksize = 64;
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
    cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

    // csr2bsr in row-major order, estimate first
    int nblocks;

    cudaMalloc((void**)&A_bsrRowPtr, sizeof(int) *(nblockrows+1));
    cusparseXcsr2bsrNnz(handle, dirA, nrows, ncols, csr_descr,
                        A_csrRowPtr, A_csrColInd, blocksize, bsr_descr, A_bsrRowPtr, &nblocks);
    cudaMalloc((void**)&A_bsrColInd, sizeof(int)*nblocks);
    printf("blocksize: %d, nblockrows: %d, nblocks: %d\n", blocksize, nblockrows, nblocks);
    unsigned bytes = (nblocks * blocksize * 8 + (nblockrows+1+nblocks) * 4) * 2;
    printf("bsr total size: "); printBytes(bytes); printf("\n");

    // free cusparse descr and handle memory
    cusparseDestroyMatDescr(csr_descr);
    cusparseDestroyMatDescr(bsr_descr);
    cusparseDestroy(handle);

    // packed matrix tA
    ullong* tA;
    cudaMalloc((void**)&tA, nblocks * blocksize * sizeof(ullong));
    csr2bsr_batch_64_row(h_csrRowPtr, h_csrColInd, nrows, ncols, A_nnz,
                     A_bsrRowPtr, A_bsrColInd, tA, blocksize, nblockrows, nblocks); //row-major

    // csr2csc for B as A^T
    int* B_cscRowInd, *B_cscColPtr;
    float* B_cscVal;
    cudaMalloc(&B_cscRowInd, sizeof(int) * A_nnz);
    cudaMalloc(&B_cscColPtr, sizeof(int) * (nrows+1));
    cudaMalloc(&B_cscVal, sizeof(float) * A_nnz);

    cusparseHandle_t handle_csr2csc;
    cusparseCreate(&handle_csr2csc);
    cusparseScsr2csc(handle_csr2csc, nrows, ncols, A_nnz,
                     A_csrVal, A_csrRowPtr, A_csrColInd,
                     B_cscVal, B_cscRowInd, B_cscColPtr,
                     CUSPARSE_ACTION_NUMERIC,
                     CUSPARSE_INDEX_BASE_ZERO);
    cusparseDestroy(handle_csr2csc);

    int *h_B_cscRowInd, *h_B_cscColPtr;
    h_B_cscRowInd = (int*) malloc(sizeof(int) * A_nnz);
    h_B_cscColPtr = (int*) malloc(sizeof(int) * (nrows+1));
    cudaMemcpy(h_B_cscRowInd, B_cscRowInd, sizeof(int) * A_nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B_cscColPtr, B_cscColPtr, sizeof(int) * (nrows+1), cudaMemcpyDeviceToHost);

    // csr2bsr for B & pack matrix for tB
    int* B_bsrRowPtr, *B_bsrColInd;
    cudaMalloc(&B_bsrRowPtr, sizeof(int) * (nblockrows+1));
    cudaMalloc(&B_bsrColInd, sizeof(int) * nblocks);
    ullong* tB;
    cudaMalloc((void**)&tB, nblocks * blocksize * sizeof(ullong));
    csr2bsr_batch_64_col(h_B_cscColPtr, h_B_cscRowInd, nrows, ncols, A_nnz,
                     B_bsrRowPtr, B_bsrColInd, tB, blocksize, nblockrows, nblocks); //col-major
    free(h_B_cscRowInd);
    free(h_B_cscColPtr);

    // ============================================= BSTC-64 bsr bmm
    // allocate bsr storage for resulting C
    // use 1 float to store the reduced sum for now
    int* fC;
	cudaMalloc((void**)&fC, sizeof(int) * 1);
	setDeviceValArr<int, int><<<1,1>>>(fC, 1, 0);

    // get grid dim
    int gridDim = (int)ceil(cbrt((double)nblockrows));
    dim3 grid(gridDim, gridDim, gridDim);

#ifdef VERBOSE
    printf("cbrt(nblockrows) = %d\n", gridDim);
#endif

int *runtime;
#ifdef PROF
    cudaMalloc(&runtime, nblockrows * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(runtime, nblockrows, 0);
#endif

    // ------

    GpuTimer bmm_timer;
    bmm_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) { // follow warp consolidation model (32 threads per block)

        bmm64_sparse<int, int><<<grid, 32>>>(tA, tB, fC,
                                               A_bsrRowPtr, A_bsrColInd,
                                               B_bsrRowPtr, B_bsrColInd,
                                               nblockrows, nblocks, nrows, runtime);
    }

    bmm_timer.Stop();
    double bmm64_time = bmm_timer.ElapsedMillis()/double(TEST_TIMES);

    // ------
#ifdef PROF
    printTimeReport<<<1,1>>>(runtime, nblockrows); cudaFree(runtime);
#endif

    int ntris_bmm;
    cudaMemcpy(&ntris_bmm, fC, sizeof(int) * 1, cudaMemcpyDeviceToHost);

    printf("ntris_bmm: %d\n", ntris_bmm);
    printf("BSR BMM-64: %.3lf\n", bmm64_time);

    // ============================================= cuSPARSE csr spgemm-float
    // setup cusparse metadata
    cusparseHandle_t handle_csr;
    cusparseCreate(&handle_csr);

    cusparseMatDescr_t A_descr;
    cusparseCreateMatDescr(&A_descr);
    cusparseSetMatType(A_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(A_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t B_descr;
    cusparseCreateMatDescr(&B_descr);
    cusparseSetMatType(B_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(B_descr, CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t C_descr;
    cusparseCreateMatDescr(&C_descr);
    cusparseSetMatType(C_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(C_descr, CUSPARSE_INDEX_BASE_ZERO);

    // setup input and output csr storage
    // 2) get B = A^T (get B = A here, and transpose using CUSPARSE_OPERATION_TRANSPOSE)
    // duplicate A as B
    int* B_csrRowPtr, *B_csrColInd;
    float* B_csrVal;
    int B_nnz = A_nnz;
    cudaMalloc(&B_csrRowPtr, sizeof(int) * (nrows+1));
    cudaMalloc(&B_csrColInd, sizeof(int) * B_nnz);
    cudaMalloc(&B_csrVal, sizeof(float) * B_nnz);
    cudaMemcpy(B_csrRowPtr, B_cscColPtr, sizeof(int) * (nrows+1), cudaMemcpyDeviceToDevice);
    cudaMemcpy(B_csrColInd, B_cscRowInd, sizeof(int) * B_nnz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(B_csrVal, B_cscVal, sizeof(float) * B_nnz, cudaMemcpyDeviceToDevice);

    // calculate nnz in C and allocate storage
    int* C_csrRowPtr, *C_csrColInd;
    float* C_csrVal;
    int C_nnz;
    cudaMalloc(&C_csrRowPtr, sizeof(int) * (nrows+1));
    cusparseXcsrgemmNnz(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, nrows, nrows, nrows,
                        A_descr, A_nnz, A_csrRowPtr, A_csrColInd,
                        B_descr, B_nnz, B_csrRowPtr, B_csrColInd,
                        C_descr, C_csrRowPtr, &C_nnz);

    cudaMalloc(&C_csrColInd, sizeof(int) * C_nnz);
    cudaMalloc(&C_csrVal, sizeof(float) * C_nnz);

    // ------

    GpuTimer csr_timer;
    csr_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) {
        cusparseScsrgemm(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, nrows, nrows, nrows,
                         A_descr, A_nnz, A_csrVal, A_csrRowPtr, A_csrColInd,
                         B_descr, B_nnz, B_csrVal, B_csrRowPtr, B_csrColInd,
                         C_descr, C_csrVal, C_csrRowPtr, C_csrColInd);
    }

    csr_timer.Stop();
    double cusparsecsrspgemmfloat_time = csr_timer.ElapsedMillis()/double(TEST_TIMES);

    // ------

    // the result include C_csrVal, C_csrRowPtr, C_csrColInd
    int* result_cusparsecsrspgemmfloat;
    cudaMalloc((void**)&result_cusparsecsrspgemmfloat, sizeof(int) * 1);
    reuduceSum<float><<<1,1>>>(C_csrVal, C_nnz, result_cusparsecsrspgemmfloat);
    int ntris_csr;
    cudaMemcpy(&ntris_csr, result_cusparsecsrspgemmfloat, sizeof(int) * 1, cudaMemcpyDeviceToHost);
    printf("ntris_csr: %d\n", ntris_csr);


    //============================================= check result
    //printf("CuSPARSE CSR SpGEMM-float (baseline) success: %d\n", check_result(result_cusparsebsrspmvfloat, result_cusparsebsrspmvfloat, ncols));
    //printf("BSR BMM-32 success: %d\n", check_result(result_bsrbmv32, result_cusparsebsrspmvfloat, ncols));

    printf("CuSPARSE CSR SpGEMM-float: %.3lf\n", cusparsecsrspgemmfloat_time);

    //============================================= free memory
    // free bsr bmm
    cudaFree(fC);

    cudaFree(tB);
    cudaFree(B_bsrColInd);
    cudaFree(B_bsrRowPtr);

    cudaFree(tA);
    cudaFree(A_bsrColInd);
    cudaFree(A_bsrRowPtr);


    // free cusparse csr spmv
    cudaFree(C_csrVal);
    cudaFree(C_csrColInd);
    cudaFree(C_csrRowPtr);
    cudaFree(B_csrVal);
    cudaFree(B_csrColInd);
    cudaFree(B_csrRowPtr);
    cudaFree(d_A_nnz);
    cudaFree(A_csrVal);
    cudaFree(A_csrColInd);
    cudaFree(A_csrRowPtr);
    cusparseDestroyMatDescr(C_descr);
    cusparseDestroyMatDescr(B_descr);
    cusparseDestroyMatDescr(A_descr);
    cusparseDestroy(handle_csr);

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
#if BLOCKSIZE == 64
    main64(argc, argv);
#elif BLOCKSIZE == 32
    main32(argc, argv);
#elif BLOCKSIZE == 16
    main16(argc, argv);
#elif BLOCKSIZE == 8
    main8(argc, argv);
#else
    main4(argc, argv);
#endif
}
