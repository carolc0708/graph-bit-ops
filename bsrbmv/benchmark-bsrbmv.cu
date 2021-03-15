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

/* bsrbmv-8 */
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
    unsigned csrbytes = (nrows+1+nnz*2) * 4;
    printf("csr total size: "); printBytes(csrbytes); printf("\n");

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

	// transform from csr to bsr using cuSPARSE
	int* bsrRowPtr, *bsrColInd;
	float* bsrVal;
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

    cudaMalloc((void**)&bsrRowPtr, sizeof(int) *(nblockrows+1));
    cusparseXcsr2bsrNnz(handle, dirA, nrows, ncols, csr_descr,
                        csrRowPtr, csrColInd, blocksize, bsr_descr, bsrRowPtr, &nblocks);
    cudaMalloc((void**)&bsrColInd, sizeof(int)*nblocks);
    printf("blocksize: %d, nblockrows: %d, nblocks: %d\n", blocksize, nblockrows, nblocks);
    unsigned bytes = (nblocks * blocksize * 1 + (nblockrows+1+nblocks) * 4);
    printf("bsr total size: "); printBytes(bytes); printf("\n");

    // packed matrix
    uchar* tA;
    cudaMalloc((void**)&tA, ceil((float)nblocks/16) * 16 * blocksize * sizeof(uchar));

#ifdef NONBATCH
    // for small matrices: csr2bsr directly
    cudaMalloc((void**)&bsrVal, sizeof(float)*(blocksize*blocksize)*nblocks);
    cusparseScsr2bsr(handle, dirA, nrows, ncols, csr_descr, csrVal,
                csrRowPtr, csrColInd, blocksize, bsr_descr, bsrVal, bsrRowPtr, bsrColInd);

    // pack A
    ToBit8Col<float><<<dim3(1, ceil((float)temp_nblocks/16)), 32>>>(bsrVal, tA, nblocks); // sparse matrix
    //printGlobalBSRBlock8<<<1,1>>>(tA, blocksize, nblocks);

    // free memory
    cudaFree(bsrVal);
#else
    // use batch transform as default
    csr2bsr_batch_8(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                     bsrRowPtr, bsrColInd, tA, blocksize, nblockrows, nblocks);
//    printGlobalBSRBlock8<<<1,1>>>(tA, blocksize, 5);
#endif

    // ============================================= input vector storage
    // generate random vector
    srand(time(0));
	float *B = (float*)malloc((nblockrows * blocksize) * 1 * sizeof(float));
	for (int i = 0; i < (nblockrows * blocksize) * 1; i ++)
    {
        float x = (float)rand() / RAND_MAX;
        if (i >= ncols) B[i] = 0;
        else B[i] = (x > 0.5) ? 1 : 0;
    }

#ifdef VERBOSE
    printf("initialize a vector with size %d x 1\n", (nblockrows * blocksize));
//    printf("orivec: \n"); printHostVec(B, (nblockrows * blocksize));
#endif

    // copy to device
	float *fB;
	cudaMalloc(&fB, (nblockrows * blocksize) * 1 * sizeof(float));
	cudaMemcpy(fB, B, (nblockrows * blocksize) * 1 * sizeof(float), cudaMemcpyHostToDevice);

    // pack B
    uchar *tB;
    cudaMalloc(&tB, ceil((float)nblockrows/4)* 4 * sizeof(uchar));
    setDeviceValArr<int, uchar><<<1,1>>>(tB, ceil((float)nblockrows/4)*4, 0);

    // get gridDim, this is to avoid nblockrows being larger than MAX_gridDim (65535?!)
    int gridDim = (int)ceil(cbrt((double)nblockrows/4));
    dim3 grid(gridDim, gridDim, gridDim);

#ifdef VERBOSE
    printf("ceil(nblockrows/4) = %d, cbrt(nblockrows/4) = %d\n", (int)ceil((double)nblockrows/4), gridDim);
#endif

    ToBit8Row<float><<<grid, 32>>>(fB, tB, nblockrows); // dense vector
//    printf("binarized vec: \n"); printBin8Vec<<<1,1>>>(tB, nblockrows);


    // ============================================= BSTC-8 bsr bmv
    // init C (result storage)
    float *fC;
    cudaMalloc(&fC, (nblockrows * blocksize) * 1 * sizeof(float));
    setDeviceValArr<int, float><<<1,1>>>(fC, nblockrows * blocksize, 0);

    int gridDim_new = (int)ceil(cbrt((double)nblockrows/128));
    dim3 grid_new(gridDim_new, gridDim_new, gridDim_new);

    int *runtime;
    int *load;
#ifdef PROF
    cudaMalloc(&runtime, nblockrows * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(runtime, nblockrows, 0);

    cudaMalloc(&load, nblockrows * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(load, nblockrows, 0);
#endif

    // ------
    GpuTimer bmv_timer;
    bmv_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) { // follow warp consolidation model (32 threads per block)

//        bmv8_sparse<int, float><<<grid, 32>>>(tA, tB, fC, bsrRowPtr, bsrColInd, nblockrows, nblocks, runtime, load);
        bmv8_sparse_sharedvector<int, float><<<grid_new, 1024>>>(tA, tB, fC, bsrRowPtr, bsrColInd, nblockrows, nblocks);
    }

    bmv_timer.Stop();
    double bmv8_time = bmv_timer.ElapsedMillis()/double(TEST_TIMES);
    // ------
#ifdef PROF
    printTimenLoadReport<<<1,1>>>(runtime, load, nblockrows); cudaFree(runtime); cudaFree(load);
#endif


    // free storage
    cudaFree(tA);
    cudaFree(tB);

    // copy result to host for verification
    float* result_bsrbmv8 = (float*)malloc(nrows * 1 * sizeof(float)); // don't care padding result
    cudaMemcpy(result_bsrbmv8, fC, nrows * 1 * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef VERBOSE
//    printf("result_bsrbmv8: \n"); printResVec<float><<<1,1>>>(fC, nrows);
    printf("bsrbmv8 nnz in vec: %d\n", countNnzinVec<float>(result_bsrbmv8, nrows));
#endif

    // ============================================= cuSPARSE csr spmv-float
    // metadata for cuSPARSE API
    cusparseHandle_t handle_csr;
    cusparseMatDescr_t mat_A;
    cusparseStatus_t cusparse_status;

    cusparseCreate(&handle_csr);
    cusparseCreateMatDescr(&mat_A);
    cusparseSetMatType(mat_A, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(mat_A, CUSPARSE_INDEX_BASE_ZERO);

    // dummy multiplication variables
    // y = α ∗ op ( A ) ∗ x + β ∗ y
#if TEST_TIMES > 1
    float alpha = 1.0, beta = 1.0;
#else
    float alpha = 1.0, beta = 0.0;
#endif

    // create dense vector storage
    float *dX, *dY;
    cudaMalloc((void**)&dX, sizeof(float)*nrows);
    cudaMemcpy(dX, B, sizeof(float)*nrows, cudaMemcpyHostToDevice);  // [nrows] to [nb * blocksize] (paddings) is not moved
    cudaMalloc((void**)&dY, sizeof(float)*nrows);
    setDeviceValArr<int, float><<<1,1>>>(dY, nrows, 0);

    // ------

    GpuTimer csr_timer;
    csr_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) {
        cusparseScsrmv(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, nrows, ncols, nnz,
                    &alpha, mat_A, csrVal, csrRowPtr, csrColInd, dX, &beta, dY);
    }

    csr_timer.Stop();
    double cusparsecsrspmvfloat_time = csr_timer.ElapsedMillis()/double(TEST_TIMES);

    // ------

    // copy result to host for verification
    float* result_cusparsecsrspmvfloat = (float*)malloc(nrows * 1 * sizeof(float));
    cudaMemcpy(result_cusparsecsrspmvfloat, dY, nrows * 1 * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef VERBOSE
//    printf("csrspmvvec: \n"); printResVec<float><<<1,1>>>(dY, nrows);
    printf("cuSPARSE nnz in vec: %d\n", countNnzinVec<float>(result_cusparsecsrspmvfloat, nrows));
#endif

    //============================================= check result
    // verify bsrbmv with cuSPARSE baseline
    printf("BSR BMV-8 success: %d\n", checkResult<float>(result_bsrbmv8, result_cusparsecsrspmvfloat, nrows));

    // print time
    printf("BSR BMV-8: %.3lf\n", bmv8_time);
    printf("CuSPARSE CSR SpMV-float: %.3lf\n", cusparsecsrspmvfloat_time);

    //============================================= free memory
    // free cusparse bsr metadata
    cusparseDestroyMatDescr(csr_descr);
    cusparseDestroyMatDescr(bsr_descr);
    cusparseDestroy(handle);

    // free cusparse csr spmv
    cusparseDestroyMatDescr(mat_A);
    cusparseDestroy(handle_csr);
    cudaFree(dX);
    cudaFree(dY);

    // free mem
    free(h_csrRowPtr);
    free(h_csrColInd);
    free(h_csrVal);

    cudaFree(csrRowPtr);
    cudaFree(csrColInd);
    cudaFree(csrVal);

    cudaFree(fB);
    cudaFree(fC);

    // free all results
    free(result_bsrbmv8);
    free(result_cusparsecsrspmvfloat);
}

/* bsrbmv-16 */
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
    unsigned csrbytes = (nrows+1+nnz*2) * 4;
    printf("csr total size: "); printBytes(csrbytes); printf("\n");

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

	// transform from csr to bsr using cuSPARSE
	int* bsrRowPtr, *bsrColInd;
	float* bsrVal;
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

    cudaMalloc((void**)&bsrRowPtr, sizeof(int) *(nblockrows+1));
    cusparseXcsr2bsrNnz(handle, dirA, nrows, ncols, csr_descr,
                        csrRowPtr, csrColInd, blocksize, bsr_descr, bsrRowPtr, &nblocks);
    cudaMalloc((void**)&bsrColInd, sizeof(int)*nblocks);
    printf("blocksize: %d, nblockrows: %d, nblocks: %d\n", blocksize, nblockrows, nblocks);
    unsigned bytes = (nblocks * blocksize * 2 + (nblockrows+1+nblocks) * 4);
    printf("bsr total size: "); printBytes(bytes); printf("\n");

    // packed matrix
    ushort* tA;
    cudaMalloc((void**)&tA, ceil((float)nblocks/4) * 4 * blocksize * sizeof(ushort)); // <--

#ifdef NONBATCH
    // for small matrices: csr2bsr directly
    cudaMalloc((void**)&bsrVal, sizeof(float)*(blocksize*blocksize)*nblocks);
    cusparseScsr2bsr(handle, dirA, nrows, ncols, csr_descr, csrVal,
                csrRowPtr, csrColInd, blocksize, bsr_descr, bsrVal, bsrRowPtr, bsrColInd);

    // pack A
    ToBit16Col<float><<<dim3(1, ceil((float)temp_nblocks/4)), 32>>>(bsrVal, tA, nblocks); // sparse matrix
    //printGlobalBSRBlock16<<<1,1>>>(tA, blocksize, nblocks);

    // free memory
    cudaFree(bsrVal);
#else
    // use batch transform as default
    csr2bsr_batch_16(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                     bsrRowPtr, bsrColInd, tA, blocksize, nblockrows, nblocks);
//    printGlobalBSRBlock16<<<1,1>>>(tA, blocksize, 5);
#endif

    // ============================================= input vector storage
    // generate random vector
    srand(time(0));
	float *B = (float*)malloc((nblockrows * blocksize) * 1 * sizeof(float));
	for (int i = 0; i < (nblockrows * blocksize) * 1; i ++)
    {
        float x = (float)rand() / RAND_MAX;
        if (i >= ncols) B[i] = 0;
        else B[i] = (x > 0.5) ? 1 : 0;
    }

#ifdef VERBOSE
    printf("initialize a vector with size %d x 1\n", (nblockrows * blocksize));
//    printf("orivec: \n"); printHostVec(B, (nblockrows * blocksize));
#endif

    // copy to device
	float *fB;
	cudaMalloc(&fB, (nblockrows * blocksize) * 1 * sizeof(float));
	cudaMemcpy(fB, B, (nblockrows * blocksize) * 1 * sizeof(float), cudaMemcpyHostToDevice);

    // pack B
    ushort *tB;
    cudaMalloc(&tB, ceil((float)nblockrows/2)* 2 * sizeof(ushort));
    setDeviceValArr<int, ushort><<<1,1>>>(tB, ceil((float)nblockrows/2)*2, 0);

    // get gridDim, this is to avoid nblockrows being larger than MAX_gridDim (65535?!)
    int gridDim = (int)ceil(cbrt((double)nblockrows/2));
    dim3 grid(gridDim, gridDim, gridDim);

#ifdef VERBOSE
    printf("ceil(nblockrows/2) = %d, cbrt(nblockrows/2) = %d\n", (int)ceil((double)nblockrows/2), gridDim);
#endif

    ToBit16Row<float><<<grid, 32>>>(fB, tB, nblockrows); // dense vector
//    printf("binarized vec: \n"); printBin16Vec<<<1,1>>>(tB, nblockrows);


    // ============================================= BSTC-16 bsr bmv
    // init C (result storage)
    float *fC;
    cudaMalloc(&fC, (nblockrows * blocksize) * 1 * sizeof(float));
    setDeviceValArr<int, float><<<1,1>>>(fC, nblockrows * blocksize, 0);

    int gridDim_new = (int)ceil(cbrt((double)nblockrows/64));
    dim3 grid_new(gridDim_new, gridDim_new, gridDim_new);

    int *runtime;
    int *load;
#ifdef PROF
    cudaMalloc(&runtime, nblockrows * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(runtime, nblockrows, 0);

    cudaMalloc(&load, nblockrows * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(load, nblockrows, 0);
#endif

    // ------
    GpuTimer bmv_timer;
    bmv_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) { // follow warp consolidation model (32 threads per block)

        //bmv16_sparse<int, float><<<grid, 32>>>(tA, tB, fC, bsrRowPtr, bsrColInd, nblockrows, nblocks, runtime, load);
        bmv16_sparse_sharedvector<int, float><<<grid_new, 1024>>>(tA, tB, fC, bsrRowPtr, bsrColInd, nblockrows, nblocks);
    }

    bmv_timer.Stop();
    double bmv16_time = bmv_timer.ElapsedMillis()/double(TEST_TIMES);
    // ------
#ifdef PROF
    printTimenLoadReport<<<1,1>>>(runtime, load, nblockrows); cudaFree(runtime); cudaFree(load);
#endif


    // free storage
    cudaFree(tA);
    cudaFree(tB);

    // copy result to host for verification
    float* result_bsrbmv16 = (float*)malloc(nrows * 1 * sizeof(float)); // don't care padding result
    cudaMemcpy(result_bsrbmv16, fC, nrows * 1 * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef VERBOSE
//    printf("result_bsrbmv16: \n"); printResVec<float><<<1,1>>>(fC, nrows);
    printf("bsrbmv16 nnz in vec: %d\n", countNnzinVec<float>(result_bsrbmv16, nrows));
#endif

    // ============================================= cuSPARSE csr spmv-float
    // metadata for cuSPARSE API
    cusparseHandle_t handle_csr;
    cusparseMatDescr_t mat_A;
    cusparseStatus_t cusparse_status;

    cusparseCreate(&handle_csr);
    cusparseCreateMatDescr(&mat_A);
    cusparseSetMatType(mat_A, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(mat_A, CUSPARSE_INDEX_BASE_ZERO);

    // dummy multiplication variables
    // y = α ∗ op ( A ) ∗ x + β ∗ y
#if TEST_TIMES > 1
    float alpha = 1.0, beta = 1.0;
#else
    float alpha = 1.0, beta = 0.0;
#endif

    // create dense vector storage
    float *dX, *dY;
    cudaMalloc((void**)&dX, sizeof(float)*nrows);
    cudaMemcpy(dX, B, sizeof(float)*nrows, cudaMemcpyHostToDevice);  // [nrows] to [nb * blocksize] (paddings) is not moved
    cudaMalloc((void**)&dY, sizeof(float)*nrows);
    setDeviceValArr<int, float><<<1,1>>>(dY, nrows, 0);

    // ------

    GpuTimer csr_timer;
    csr_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) {
        cusparseScsrmv(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, nrows, ncols, nnz,
                    &alpha, mat_A, csrVal, csrRowPtr, csrColInd, dX, &beta, dY);
    }

    csr_timer.Stop();
    double cusparsecsrspmvfloat_time = csr_timer.ElapsedMillis()/double(TEST_TIMES);

    // ------

    // copy result to host for verification
    float* result_cusparsecsrspmvfloat = (float*)malloc(nrows * 1 * sizeof(float));
    cudaMemcpy(result_cusparsecsrspmvfloat, dY, nrows * 1 * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef VERBOSE
//    printf("csrspmvvec: \n"); printResVec<float><<<1,1>>>(dY, nrows);
    printf("cuSPARSE nnz in vec: %d\n", countNnzinVec<float>(result_cusparsecsrspmvfloat, nrows));
#endif

    //============================================= check result
    // verify bsrbmv with cuSPARSE baseline
    printf("BSR BMV-16 success: %d\n", checkResult<float>(result_bsrbmv16, result_cusparsecsrspmvfloat, nrows));

    // print time
    printf("BSR BMV-16: %.3lf\n", bmv16_time);
    printf("CuSPARSE CSR SpMV-float: %.3lf\n", cusparsecsrspmvfloat_time);

    //============================================= free memory
    // free cusparse bsr metadata
    cusparseDestroyMatDescr(csr_descr);
    cusparseDestroyMatDescr(bsr_descr);
    cusparseDestroy(handle);

    // free cusparse csr spmv
    cusparseDestroyMatDescr(mat_A);
    cusparseDestroy(handle_csr);
    cudaFree(dX);
    cudaFree(dY);

    // free mem
    free(h_csrRowPtr);
    free(h_csrColInd);
    free(h_csrVal);

    cudaFree(csrRowPtr);
    cudaFree(csrColInd);
    cudaFree(csrVal);

    cudaFree(fB);
    cudaFree(fC);

    // free all results
    free(result_bsrbmv16);
    free(result_cusparsecsrspmvfloat);
}

/* bsrbmv-32 */
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
    unsigned csrbytes = (nrows+1+nnz*2) * 4;
    printf("csr total size: "); printBytes(csrbytes); printf("\n");

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

	// transform from csr to bsr using cuSPARSE
	int* bsrRowPtr, *bsrColInd;
	float* bsrVal;
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

    cudaMalloc((void**)&bsrRowPtr, sizeof(int) *(nblockrows+1));
    cusparseXcsr2bsrNnz(handle, dirA, nrows, ncols, csr_descr,
                        csrRowPtr, csrColInd, blocksize, bsr_descr, bsrRowPtr, &nblocks);
    cudaMalloc((void**)&bsrColInd, sizeof(int)*nblocks);
    printf("blocksize: %d, nblockrows: %d, nblocks: %d\n", blocksize, nblockrows, nblocks);
    unsigned bytes = (nblocks * blocksize * 4 + (nblockrows+1+nblocks) * 4);
    printf("bsr total size: "); printBytes(bytes); printf("\n");

    // packed matrix
    unsigned* tA;
    cudaMalloc((void**)&tA, nblocks * blocksize * sizeof(unsigned));

#ifdef NONBATCH
    // for small matrices: csr2bsr directly
    cudaMalloc((void**)&bsrVal, sizeof(float)*(blocksize*blocksize)*nblocks);
    cusparseScsr2bsr(handle, dirA, nrows, ncols, csr_descr, csrVal,
                csrRowPtr, csrColInd, blocksize, bsr_descr, bsrVal, bsrRowPtr, bsrColInd);

    // pack A
    ToBit32Col<float><<<dim3(1, nblocks), 32>>>(bsrVal, tA, blocksize, nblocks * blocksize); // sparse matrix
//    printGlobalBSRBlock32<<<1,1>>>(tA, blocksize, nblocks);

    // free memory
    cudaFree(bsrVal);
#else
    // use batch transform as default
    csr2bsr_batch_32(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                     bsrRowPtr, bsrColInd, tA, blocksize, nblockrows, nblocks);
#endif

    // ============================================= input vector storage
    // generate random vector
    srand(time(0));
	float *B = (float*)malloc((nblockrows * blocksize) * 1 * sizeof(float));
	for (int i = 0; i < (nblockrows * blocksize) * 1; i ++)
    {
        float x = (float)rand() / RAND_MAX;
        if (i >= ncols) B[i] = 0;
        else B[i] = (x > 0.5) ? 1 : 0;
    }

#ifdef VERBOSE
    printf("initialize a vector with size %d x 1\n", (nblockrows * blocksize));
//    printf("orivec: \n"); printHostVec(B, (nblockrows * blocksize));
#endif

    // copy to device
	float *fB;
	cudaMalloc(&fB, (nblockrows * blocksize) * 1 * sizeof(float));
	cudaMemcpy(fB, B, (nblockrows * blocksize) * 1 * sizeof(float), cudaMemcpyHostToDevice);

    // pack B
    unsigned *tB;
    cudaMalloc(&tB, nblockrows * 1 * sizeof(unsigned)); // (nblockrows * blocksize) / 32 = nblockrows

    // get gridDim, this is to avoid nblockrows being larger than MAX_gridDim (65535?!)
    int gridDim = (int)ceil(cbrt((double)nblockrows));
    dim3 grid(gridDim, gridDim, gridDim);

#ifdef VERBOSE
    printf("cbrt(nblockrows) = %d\n", gridDim);
#endif

    ToBit32Row<float><<<grid, 32>>>(fB, tB, nblockrows * blocksize, 1, nblockrows); // dense vector


    // ============================================= BSTC-32 bsr bmv
    // init C (result storage)
    float *fC;
    cudaMalloc(&fC, (nblockrows * blocksize) * 1 * sizeof(float));
    setDeviceValArr<int, float><<<1,1>>>(fC, nblockrows * blocksize, 0);

    int *runtime;
    int *load;
#ifdef PROF
    cudaMalloc(&runtime, nblockrows * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(runtime, nblockrows, 0);

    cudaMalloc(&load, nblockrows * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(load, nblockrows, 0);
#endif

    // ------
    GpuTimer bmv_timer;
    bmv_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) { // follow warp consolidation model (32 threads per block)

        bmv32_sparse<int, float><<<grid, 32>>>(tA, tB, fC, blocksize, nblocks, 1, bsrRowPtr, bsrColInd, nblockrows, nblocks, runtime, load);
    }

    bmv_timer.Stop();
    double bmv32_time = bmv_timer.ElapsedMillis()/double(TEST_TIMES);
    // ------
#ifdef PROF
    printTimenLoadReport<<<1,1>>>(runtime, load, nblockrows); cudaFree(runtime); cudaFree(load);
#endif


    // free storage
    cudaFree(tA);
    cudaFree(tB);

    // copy result to host for verification
    float* result_bsrbmv32 = (float*)malloc(nrows * 1 * sizeof(float)); // don't care padding result
    cudaMemcpy(result_bsrbmv32, fC, nrows * 1 * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef VERBOSE
//    printf("result_bsrbmv32: \n"); printResVec<float><<<1,1>>>(fC, nrows);
    printf("bsrbmv32 nnz in vec: %d\n", countNnzinVec<float>(result_bsrbmv32, nrows));
#endif

    // ============================================= cuSPARSE csr spmv-float
    // metadata for cuSPARSE API
    cusparseHandle_t handle_csr;
    cusparseMatDescr_t mat_A;
    cusparseStatus_t cusparse_status;

    cusparseCreate(&handle_csr);
    cusparseCreateMatDescr(&mat_A);
    cusparseSetMatType(mat_A, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(mat_A, CUSPARSE_INDEX_BASE_ZERO);

    // dummy multiplication variables
    // y = α ∗ op ( A ) ∗ x + β ∗ y
#if TEST_TIMES > 1
    float alpha = 1.0, beta = 1.0;
#else
    float alpha = 1.0, beta = 0.0;
#endif

    // create dense vector storage
    float *dX, *dY;
    cudaMalloc((void**)&dX, sizeof(float)*nrows);
    cudaMemcpy(dX, B, sizeof(float)*nrows, cudaMemcpyHostToDevice);  // [nrows] to [nb * blocksize] (paddings) is not moved
    cudaMalloc((void**)&dY, sizeof(float)*nrows);
    setDeviceValArr<int, float><<<1,1>>>(dY, nrows, 0);

    // ------

    GpuTimer csr_timer;
    csr_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) {
        cusparseScsrmv(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, nrows, ncols, nnz,
                    &alpha, mat_A, csrVal, csrRowPtr, csrColInd, dX, &beta, dY);
    }

    csr_timer.Stop();
    double cusparsecsrspmvfloat_time = csr_timer.ElapsedMillis()/double(TEST_TIMES);

    // ------

    // copy result to host for verification
    float* result_cusparsecsrspmvfloat = (float*)malloc(nrows * 1 * sizeof(float));
    cudaMemcpy(result_cusparsecsrspmvfloat, dY, nrows * 1 * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef VERBOSE
//    printf("csrspmvvec: \n"); printResVec<float><<<1,1>>>(dY, nrows);
    printf("cuSPARSE nnz in vec: %d\n", countNnzinVec<float>(result_cusparsecsrspmvfloat, nrows));
#endif

    //============================================= check result
    // verify bsrbmv with cuSPARSE baseline
    printf("BSR BMV-32 success: %d\n", checkResult<float>(result_bsrbmv32, result_cusparsecsrspmvfloat, nrows));

    // print time
    printf("BSR BMV-32: %.3lf\n", bmv32_time);
    printf("CuSPARSE CSR SpMV-float: %.3lf\n", cusparsecsrspmvfloat_time);

    //============================================= free memory
    // free cusparse bsr metadata
    cusparseDestroyMatDescr(csr_descr);
    cusparseDestroyMatDescr(bsr_descr);
    cusparseDestroy(handle);

    // free cusparse csr spmv
    cusparseDestroyMatDescr(mat_A);
    cusparseDestroy(handle_csr);
    cudaFree(dX);
    cudaFree(dY);

    // free mem
    free(h_csrRowPtr);
    free(h_csrColInd);
    free(h_csrVal);

    cudaFree(csrRowPtr);
    cudaFree(csrColInd);
    cudaFree(csrVal);

    cudaFree(fB);
    cudaFree(fC);

    // free all results
    free(result_bsrbmv32);
    free(result_cusparsecsrspmvfloat);
}

/* bsrbmv-64 */
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
    unsigned csrbytes = (nrows+1+nnz*2) * 4;
    printf("csr total size: "); printBytes(csrbytes); printf("\n");

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

	// transform from csr to bsr using cuSPARSE
	int* bsrRowPtr, *bsrColInd;
	float* bsrVal;
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

    // csr2bsr in row-major order, estimate nblocks first
    int nblocks;

    cudaMalloc((void**)&bsrRowPtr, sizeof(int) *(nblockrows+1));
    cusparseXcsr2bsrNnz(handle, dirA, nrows, ncols, csr_descr,
                        csrRowPtr, csrColInd, blocksize, bsr_descr, bsrRowPtr, &nblocks);
    cudaMalloc((void**)&bsrColInd, sizeof(int)*nblocks);
    printf("blocksize: %d, nblockrows: %d, nblocks: %d\n", blocksize, nblockrows, nblocks);
    unsigned bytes = (nblocks * blocksize * 8 + (nblockrows+1+nblocks) * 4);
    printf("bsr total size: "); printBytes(bytes); printf("\n");

    // packed matrix tA
    ullong* tA;
    cudaMalloc((void**)&tA, nblocks * blocksize * sizeof(ullong));


#ifdef NONBATCH
    // for small matrices: csr2bsr directly
    cudaMalloc((void**)&bsrVal, sizeof(float)*(blocksize*blocksize)*nblocks);
    cusparseScsr2bsr(handle, dirA, nrows, ncols, csr_descr, csrVal,
                csrRowPtr, csrColInd, blocksize, bsr_descr, bsrVal, bsrRowPtr, bsrColInd);

    // pack A
    ToBit64Col<float><<<dim3(1, nblocks), 32>>>(bsrVal, tA, blocksize, nblocks * blocksize); // sparse matrix
//    printGlobalBSRBlock64<<<1,1>>>(tA, blocksize, nblocks);

    // free memory
    cudaFree(bsrVal);
#else
    // use batch transform as default
    csr2bsr_batch_64(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                     bsrRowPtr, bsrColInd, tA, blocksize, nblockrows, nblocks);
#endif

    // ============================================= input vector storage
    // generate random vector
    srand(time(0));
	float *B = (float*)malloc((nblockrows * blocksize) * 1 * sizeof(float));
	for (int i = 0; i < (nblockrows * blocksize) * 1; i ++)
    {
        float x = (float)rand() / RAND_MAX;
        if (i >= ncols) B[i] = 0;
        else B[i] = (x > 0.5) ? 1 : 0;
    }

#ifdef VERBOSE
    printf("initialize a vector with size %d x 1\n", (nblockrows * blocksize));
//    printf("orivec: \n"); printHostVec(B, (nblockrows * blocksize));
#endif

    // copy to device
	float *fB;
	cudaMalloc(&fB, (nblockrows * blocksize) * 1 * sizeof(float));
	cudaMemcpy(fB, B, (nblockrows * blocksize) * 1 * sizeof(float), cudaMemcpyHostToDevice);

    // pack B
    ullong *tB;
    cudaMalloc(&tB, nblockrows * 1 * sizeof(ullong)); // (nblockrows * blocksize) / 64 = nblockrows

    // get gridDim, this is to avoid nblockrows being larger than MAX_gridDim (65535?!)
    int gridDim = (int)ceil(cbrt((double)nblockrows));
    dim3 grid(gridDim, gridDim, gridDim);

#ifdef VERBOSE
    printf("cbrt(nblockrows) = %d\n", gridDim);
#endif

    ToBit64Row<float><<<grid, 32>>>(fB, tB, nblockrows * blocksize, 1, nblockrows); // dense vector


    // ============================================= BSTC-64 bsr bmv
    // init C (output storage)
    float *fC;
    cudaMalloc(&fC, (nblockrows * blocksize) * 1 * sizeof(float));
    setDeviceValArr<int, float><<<1,1>>>(fC, nblockrows * blocksize, 0);

    int *runtime;
#ifdef PROF
    cudaMalloc(&runtime, nblockrows * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(runtime, nblockrows, 0);
#endif

    // ------

    GpuTimer bmv_timer;
    bmv_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) { // follow warp consolidation model (32 threads per block)

        bmv64_sparse<int, float><<<grid, 32>>>(tA, tB, fC, blocksize, nblocks, 1, bsrRowPtr, bsrColInd, nblockrows, nblocks, runtime);
    }

    bmv_timer.Stop();
    double bmv64_time = bmv_timer.ElapsedMillis()/double(TEST_TIMES);

    // ------

#ifdef PROF
    printTimeReport<<<1,1>>>(runtime, nblockrows); cudaFree(runtime);
#endif

    // free memory
    cudaFree(tA);
    cudaFree(tB);

    // copy result to host for verification
    float* result_bsrbmv64 = (float*)malloc(nrows * 1 * sizeof(float)); // don't care padding result
    cudaMemcpy(result_bsrbmv64, fC, nrows * 1 * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef VERBOSE
//    printf("result_bsrbmv32: \n"); printResVec<float><<<1,1>>>(fC, nrows);
    printf("bsrbmv64 nnz in vec: %d\n", countNnzinVec<float>(result_bsrbmv64, nrows));
#endif

    // ============================================= cuSPARSE csr spmv-float
    // metadata for cuSPARSE API
    cusparseHandle_t handle_csr;
    cusparseMatDescr_t mat_A;
    cusparseStatus_t cusparse_status;

    cusparseCreate(&handle_csr);
    cusparseCreateMatDescr(&mat_A);
    cusparseSetMatType(mat_A, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(mat_A, CUSPARSE_INDEX_BASE_ZERO);

    // dummy multiplication variables
    // y = α ∗ op ( A ) ∗ x + β ∗ y
#if TEST_TIMES > 1
    float alpha = 1.0, beta = 1.0;
#else
    float alpha = 1.0, beta = 0.0;
#endif

    // create dense vector
    float *dX, *dY;
    cudaMalloc((void**)&dX, sizeof(float)*nrows);
    cudaMemcpy(dX, B, sizeof(float)*nrows, cudaMemcpyHostToDevice);  // [nrows] to [nb * blocksize] (paddings) is not moved
    cudaMalloc((void**)&dY, sizeof(float)*nrows);
    setDeviceValArr<int, float><<<1,1>>>(dY, nrows, 0);

    // ------
    GpuTimer csr_timer;
    csr_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) {
        cusparseScsrmv(handle_csr, CUSPARSE_OPERATION_NON_TRANSPOSE, nrows, ncols, nnz,
                    &alpha, mat_A, csrVal, csrRowPtr, csrColInd, dX, &beta, dY);
    }

    csr_timer.Stop();
    double cusparsecsrspmvfloat_time = csr_timer.ElapsedMillis()/double(TEST_TIMES);

    // ------

    // copy result to host for verification
    float* result_cusparsecsrspmvfloat = (float*)malloc(nrows * 1 * sizeof(float));
    cudaMemcpy(result_cusparsecsrspmvfloat, dY, nrows * 1 * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef VERBOSE
//    printf("csrspmvvec: \n"); printResVec<float><<<1,1>>>(dY, nrows);
    printf("cuSPARSE nnz in vec: %d\n", countNnzinVec<float>(result_cusparsecsrspmvfloat, nrows));
#endif

    //============================================= check result
    printf("BSR BMV-64 success: %d\n", checkResult<float>(result_bsrbmv64, result_cusparsecsrspmvfloat, nrows));

    printf("BSR BMV-64: %.3lf\n", bmv64_time);
    printf("CuSPARSE CSR SpMV-float: %.3lf\n", cusparsecsrspmvfloat_time);


    //============================================= free memory
    // free descr and handle memory
    cusparseDestroyMatDescr(csr_descr);
    cusparseDestroyMatDescr(bsr_descr);
    cusparseDestroy(handle);

    // free cusparse csr spmv
    cusparseDestroyMatDescr(mat_A);
    cusparseDestroy(handle_csr);
    cudaFree(dX);
    cudaFree(dY);

    // free mem
    free(h_csrRowPtr);
    free(h_csrColInd);
    free(h_csrVal);

    cudaFree(csrRowPtr);
    cudaFree(csrColInd);
    cudaFree(csrVal);

    cudaFree(fB);
    cudaFree(fC);

    // free all results
    free(result_bsrbmv64);
    free(result_cusparsecsrspmvfloat);

}

int main(int argc, char* argv[])
{
#if BLOCKSIZE == 64
    main64(argc, argv);
#elif BLOCKSIZE == 32
    main32(argc, argv);
#elif BLOCKSIZE == 16
    main16(argc, argv);
#else
    main8(argc, argv);
#endif
}
