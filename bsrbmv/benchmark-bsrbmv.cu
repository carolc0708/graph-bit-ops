#include <iostream>
#include <sys/time.h>

#define TEST_TIMES 1 // do not test more than 1 time for now (vector not clean)
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

//#include <cuda_runtime_api.h>
//#include <cusparse.h>

#include "bsrbmv.cu"
#include "mmio_highlevel.h"

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

// to print unsigned
void bin(unsigned n)
{
    unsigned i;
    for (i = 1 << 31; i > 0; i = i / 2)
        (n & i) ? printf("1") : printf("0");
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

void printind(int* indarr, const int N)
{
    for(int i=0; i<N; i++) {
        printf("%d ", indarr[i]);
    }
    printf("\n");
}

int main(int argc, char* argv[])
{

    bool trans_A = false;
    bool trans_B = false;

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

	// read bsr from file or transform csr into bsr
	// 1) read bsr data from file


	// 2) transform from csr to bsr using cuSPARSE
	int* bsrRowPtr, *bsrColInd;
	float* bsrVal;
	int blocksize = 32;

	// create cusparsematdescr for csr, bsr
    cusparseMatDescr_t csr_descr = 0;
    cusparseCreateMatDescr(&csr_descr);
    cusparseSetMatType(csr_descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(csr_descr,CUSPARSE_INDEX_BASE_ZERO);

    cusparseMatDescr_t bsr_descr = 0;
    cusparseCreateMatDescr(&bsr_descr);
    cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO);

    // cusparse handle
    cudaStream_t streamId = 0;
    cusparseHandle_t handle = 0;

    cusparseCreate(&handle);
    cusparseSetStream(handle, streamId);

    // cusparse direction
    cusparseDirection_t dirA = CUSPARSE_DIRECTION_COLUMN;

    // transform CSR to BSR with column-major order
    int base, nblocks;

    int mb = (nrows + blocksize-1)/blocksize;
    int nb = (ncols + blocksize-1)/blocksize;
    int nblockrows = mb;

    cudaMalloc((void**)&bsrRowPtr, sizeof(int) *(nblockrows+1));
    cusparseXcsr2bsrNnz(handle, dirA, nrows, ncols, csr_descr,
                        csrRowPtr, csrColInd, blocksize, bsr_descr, bsrRowPtr, &nblocks);
    cudaMalloc((void**)&bsrColInd, sizeof(int)*nblocks);
    cudaMalloc((void**)&bsrVal, sizeof(float)*(blocksize*blocksize)*nblocks);
    cusparseScsr2bsr(handle, dirA, nrows, ncols, csr_descr, csrVal,
                    csrRowPtr, csrColInd, blocksize, bsr_descr, bsrVal, bsrRowPtr, bsrColInd);
    printf("blocksize: %d, nblockrows: %d, nblocks: %d\n", blocksize, nblockrows, nblocks);
    float* h_bsrVal = (float*)malloc(sizeof(float)*blocksize*blocksize*nblocks);
    cudaMemcpy(h_bsrVal, bsrVal, sizeof(float)*blocksize*blocksize*nblocks, cudaMemcpyDeviceToHost);
    printmat(h_bsrVal, nblocks, blocksize);
    free(h_bsrVal);
//
//    int* h_bsrRowPtr = (int*)malloc(sizeof(int) *(nblockrows+1));
//    cudaMemcpy(h_bsrRowPtr, bsrRowPtr, sizeof(int) *(nblockrows+1), cudaMemcpyDeviceToHost);
//    printf("rowptr: \n"); printind(h_bsrRowPtr, (nblockrows+1));
//    free(h_bsrRowPtr);
//
//    int* h_bsrColInd = (int*)malloc(sizeof(int) *nblocks);
//    cudaMemcpy(h_bsrColInd, bsrColInd, sizeof(int) * nblocks, cudaMemcpyDeviceToHost);
//    printf("colind: \n"); printind(h_bsrColInd, nblocks);
//    free(h_bsrColInd);


    // input vector and result vector storage -----------------------------------
    // generate random vector
    srand(time(0));
	float *B = (float*)malloc((nblockrows * blocksize) * 1 * sizeof(float));
	for (int i = 0; i < (nblockrows * blocksize) * 1; i ++)
    {
        float x = (float)rand() / RAND_MAX;
        if (i >= ncols) B[i] = 0;
        else B[i] = (x > 0.5) ? 1 : 0;
    }
    printf("initialize a vector with size %d x 1\n", (nblockrows * blocksize));
    printf("orivec: \n"); printvec(B, (nblockrows * blocksize));

    // copy to cuda
	float *fB, *fC;
    unsigned *uC;
    ullong *ullC;

	cudaMalloc(&fB, (nblockrows * blocksize) * 1 * sizeof(float));
	cudaMalloc(&fC, (nblockrows * blocksize) * 1 * sizeof(float));

	cudaMalloc(&uC, (nblockrows * blocksize) * 1 * sizeof(unsigned));
	cudaMalloc(&ullC, (nblockrows * blocksize) * 1 * sizeof(unsigned long long));

	cudaMemcpy(fB, B, (nblockrows * blocksize) * 1 * sizeof(float), cudaMemcpyHostToDevice);

	// time measurement setup -----------------------------------
	cudaEvent_t start, stop;
	float milliseconds = 0;

    // ============================================= BSTC-32 bsr bmv
    cudaMemset(fC, 0, (nblockrows * blocksize) * 1 * sizeof(float));

    unsigned *tA, *tB;
	cudaMalloc(&tA, nblocks * blocksize * sizeof(unsigned)); // (nblocks * blocksize * blocksize) / 32 = nblocks * blocksize
	cudaMalloc(&tB, nblockrows * 1 * sizeof(unsigned)); // (nblockrows * blocksize) / 32 = nblockrows
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ------
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++) { // follow warp consolidation model (32 threads per block)
        ToBit32Col<float><<<dim3(CEIL(blocksize), CEIL(nblocks * blocksize)), 32>>>(bsrVal, tA, blocksize, nblocks * blocksize); // sparse matrix

//        unsigned* h_tA = (unsigned*) malloc(nblocks * blocksize * sizeof(unsigned));
//        cudaMemcpy(h_tA, tA, nblocks * blocksize * sizeof(unsigned), cudaMemcpyDeviceToHost);
//        printf("binmat:\n"); printbinmat(h_tA, nblocks, blocksize);

        ToBit32Row<float><<<dim3(CEIL(nblockrows * blocksize), CEIL(1)), 32>>>(fB, tB, nblockrows * blocksize, 1); // dense vector

//        unsigned* h_tB = (unsigned*) malloc(nblockrows * sizeof(unsigned));
//        cudaMemcpy(h_tB, tB, nblockrows * sizeof(unsigned), cudaMemcpyDeviceToHost);
//        printf("binvec:\n"); printbinvec(h_tB, nblockrows);

        bmv32_sparse<int, float><<<dim3(CEIL(1), CEIL(nblockrows * blocksize)), 32>>>(tA, tB, fC, blocksize, nblocks, 1,
                                                                                    bsrRowPtr, bsrColInd, nblockrows, nblocks);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double bmv32_time = (milliseconds*1e3)/double(TEST_TIMES);

    cudaFree(tA);
    cudaFree(tB);
    // ------

    float* result_bsrbmv32 = (float*)malloc(ncols * 1 * sizeof(float)); // don't care padding result
    cudaMemcpy(result_bsrbmv32, fC, ncols * 1 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("resultvec: \n"); printresvec(result_bsrbmv32, ncols);


    // ============================================= cuSPARSE bsr spmv-float 32
    // y = α ∗ op ( A ) ∗ x + β ∗ y
    // allocate vector x and vector y large enough for bsrmv
    float *x, *y;
    cudaMalloc((void**)&x, sizeof(float)*(nblockrows * blocksize));
    cudaMemcpy(x, fB, sizeof(float)*ncols, cudaMemcpyHostToDevice);  // [ncols] to [nb * blocksize] (paddings) is not moved
    cudaMalloc((void**)&y, sizeof(float)*(nblockrows * blocksize));
    cudaMemset(y, 0, sizeof(float)*ncols);

    // perform bsrmv
    float alpha = 1.0, beta = 0.0;
    cusparseOperation_t transA = CUSPARSE_OPERATION_NON_TRANSPOSE;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // ------
    cudaEventRecord(start);
    for (int i=0; i<TEST_TIMES; i++) {
        cusparseSbsrmv(handle, dirA, transA, mb, nb, nblocks, &alpha,
                    bsr_descr, bsrVal, bsrRowPtr, bsrColInd, blocksize, x, &beta, y);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds,start,stop);
    double cusparsebsrspmvfloat_time = (milliseconds*1e3)/double(TEST_TIMES);
    // ------

    float* result_cusparsebsrspmvfloat = (float*)malloc(ncols * 1 * sizeof(float));
    cudaMemcpy(result_cusparsebsrspmvfloat, y, ncols * 1 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("baselinevec: \n"); printresvec(result_cusparsebsrspmvfloat, ncols);

//    // ============================================= cuSPARSE csr spmv-float
//    // CUSPARSE APIs
//    cusparseSpMatDescr_t matA;
//    cusparseDnVecDescr_t vecX, vecY;
//    void* dBuffer = NULL;
//    size_t bufferSize = 0;
//
//    // Create sparse matrix A in CSR format
//    cusparseCreateCsr(&matA, nrows, ncols, nnz, csrRowPtr, csrColInd, csrVal,
//                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
//
//    // create dense vector
//    float *dX, *dY;
//    cudaMalloc((void**)&dX, sizeof(float)*(nblockrows * blocksize));
//    cudaMemcpy(dX, fB, sizeof(float)*ncols, cudaMemcpyHostToDevice);  // [ncols] to [nb * blocksize] (paddings) is not moved
//    cudaMalloc((void**)&dY, sizeof(float)*(nblockrows * blocksize));
//    cudaMemset(dY, 0, sizeof(float)*ncols);
//
//    // Create dense vector X
//    cusparseCreateDnVec(&vecX, ncols, dX, CUDA_R_32F);
//    // Create dense vector y
//    cusparseCreateDnVec(&vecY, nrows, dY, CUDA_R_32F);
//
//    // allocate an external buffer if needed
//    cusparseSpMV_bufferSize(handle, transA, &alpha, matA, vecX, &beta, vecY,
//                            CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
//    cudaMalloc(&dBuffer, bufferSize);
//
//    // execute SpMV
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//    // ------
//    cudaEventRecord(start);
//    for (int i=0; i<TEST_TIMES; i++) {
//        cusparseSpMV(handle, transA, &alpha, matA, vecX, &beta, vecY,
//                        CUDA_R_32F, CUSPARSE_MV_ALG_DEFAULT, dBuffer);
//    }
//    cudaEventRecord(stop);
//    cudaEventSynchronize(stop);
//
//    milliseconds = 0;
//    cudaEventElapsedTime(&milliseconds,start,stop);
//    double cusparsecsrspmvfloat_time = (milliseconds*1e3)/double(TEST_TIMES);
//    // ------
//
//    float* result_cusparsecsrspmvfloat = (float*)malloc(ncols * 1 * sizeof(float));
//    cudaMemcpy(result_cusparsecsrspmvfloat, vecY, ncols * 1 * sizeof(float), cudaMemcpyDeviceToHost);


    // ============================================= BSTC-64 bsr bmv

    // ============================================= cuSPARSE bsr spmv-float 64

    // ============================================= BSTC-32 bsr bmm

    // ============================================= cuSPARSE csr spgemm

    //============================================= CHECK RESULT
    //printf("CuSPARSE BSR SpMV-float (baseline) success: %d\n", check_result(result_cusparsebsrspmvfloat, result_cusparsebsrspmvfloat, ncols));
    //printf("BSR BMV-32 success: %d\n", check_result(result_bsrbmv32, result_cusparsebsrspmvfloat, ncols));

    printf("CuSPARSE BSR SpMV-float: %.3lf\n", cusparsebsrspmvfloat_time);
    printf("BSR BMV-32: %.3lf\n", bmv32_time);
    //printf("CuSPARSE CSR SpMV-float: %.3lf\n", cusparsecsrspmvfloat_time);

    // free descr and handle memory
    cusparseDestroyMatDescr(csr_descr);
    csr_descr = 0;
    cusparseDestroyMatDescr(bsr_descr);
    bsr_descr = 0;
    cusparseDestroy(handle);
    handle = 0;

    // free cusparse bsr spmv
    cudaFree(x);
    cudaFree(y);

//    // free cusparse csr spmv
//    cusparseDestroySpMat(matA);
//    cusparseDestroyDnVec(vecX);
//    cusparseDestroyDnVec(vecY);
//    cudaFree(dX);
//    cudaFree(dY);

    // free mem
    free(h_csrRowPtr);
    free(h_csrColInd);
    free(h_csrVal);

    cudaFree(csrRowPtr);
    cudaFree(csrColInd);
    cudaFree(csrVal);

    cudaFree(fB);
    cudaFree(fC);
    cudaFree(uC);
    cudaFree(ullC);

    // free all results
    free(result_cusparsebsrspmvfloat);
    free(result_bsrbmv32);
    //free(result_cusparsecsrspmvfloat);

}
