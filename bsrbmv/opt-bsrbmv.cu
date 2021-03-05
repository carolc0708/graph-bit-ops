#include <iostream>
#include <sys/time.h>

#define TEST_TIMES 5
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#include "mmio_highlevel.h"
#include "csr2bsr_batch.cu"

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
    printf("initialize a vector with size %d x 1\n", (nblockrows * blocksize));
//    printf("orivec: \n"); printHostVec(B, (nblockrows * blocksize));

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
    printf("cbrt(nblockrows) = %d\n", gridDim);

    ToBit32Row<float><<<grid, 32>>>(fB, tB, nblockrows * blocksize, 1, nblockrows); // dense vector


    // ============================================= BSTC-32 bsr bmv
    // init C (result storage)
    float *fC;
    cudaMalloc(&fC, (nblockrows * blocksize) * 1 * sizeof(float));
    setDeviceValArr<int, float><<<1,1>>>(fC, nblockrows * blocksize, 0);

    //  ===== configure workload =====

    int MAX = atoi(argv[2]);

    // set grid size

    int gridDim_ws = (int)ceil(cbrt((double)nblocks/MAX));
    dim3 grid_ws(gridDim_ws, gridDim_ws, gridDim_ws);
    printf("nblocks: %d, MAX: %d\n", nblocks, MAX);
    printf("workload unit ceil(nblocks/MAX) = %d\n", (int)ceil((double)nblocks/MAX));
    printf("cbrt(nblocks/MAX) = %d\n", gridDim_ws);

    // bsr2bcoo
    int* rowind;
    cudaMalloc((void**)&rowind, nblocks * sizeof(int));
    bsr2bcoo<<<1,1>>>(bsrRowPtr, nblockrows, bsrColInd, rowind);
//    printResVec<int><<<1,1>>>(rowind, nblocks);


    //  ===== configure workload =====

    int *runtime;
    int *load;
#ifdef PROF
    cudaMalloc(&runtime, workloadsize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(runtime, workloadsize, 0);

    cudaMalloc(&load, nblockrows * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(load, workloadsize, 0);
#endif

    // ------
    GpuTimer bmv_timer;
    bmv_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) { // follow warp consolidation model (32 threads per block)

        bmv32_sparse_opt<int, float><<<grid_ws, 32>>>(tA, tB, fC, rowind, bsrColInd, nblocks, MAX, runtime, load);
    }

    bmv_timer.Stop();
    double bmv32_time = bmv_timer.ElapsedMillis()/double(TEST_TIMES);
    // ------
#ifdef PROF
    printTimenLoadReport<<<1,1>>>(runtime, load, workloadsize); cudaFree(runtime); cudaFree(load);
#endif


    // free storage
    cudaFree(tA);
    cudaFree(tB);

    // copy result to host for verification
    float* result_bsrbmv32 = (float*)malloc(nrows * 1 * sizeof(float)); // don't care padding result
    cudaMemcpy(result_bsrbmv32, fC, nrows * 1 * sizeof(float), cudaMemcpyDeviceToHost);
//    printf("result_bsrbmv32: \n"); printResVec<float><<<1,1>>>(fC, nrows);
    printf("bsrbmv32 nnz in vec: %d\n", countNnzinVec<float>(result_bsrbmv32, nrows));

//    cudaFree(workloadptr);


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
//    printf("csrspmvvec: \n"); printResVec<float><<<1,1>>>(dY, nrows);
    printf("cuSPARSE nnz in vec: %d\n", countNnzinVec<float>(result_cusparsecsrspmvfloat, nrows));

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
    printf("initialize a vector with size %d x 1\n", (nblockrows * blocksize));
//    printf("orivec: \n"); printHostVec(B, (nblockrows * blocksize));

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
    printf("cbrt(nblockrows) = %d\n", gridDim);

    ToBit64Row<float><<<grid, 32>>>(fB, tB, nblockrows * blocksize, 1, nblockrows); // dense vector


    // ============================================= BSTC-64 bsr bmv
    // init C (output storage)
    float *fC;
    cudaMalloc(&fC, (nblockrows * blocksize) * 1 * sizeof(float));
    setDeviceValArr<int, float><<<1,1>>>(fC, nblockrows * blocksize, 0);

    //  ===== configure workload =====
//    int MIN = 10;
//    int *workloadsizeptr;
//    cudaMalloc((void**)&workloadsizeptr, 1 * sizeof(int));
//    count_workload_split<<<1,1>>>(workloadsizeptr, bsrRowPtr, nblockrows, bsrColInd, MIN);
//
//    int workloadsize;
//    cudaMemcpy(&workloadsize, workloadsizeptr, sizeof(int) * 1, cudaMemcpyDeviceToHost);
//    printf("workloadsize: %d (nblockrows: %d)\n", workloadsize, nblockrows);
//
//    int *workloadptr;
//    cudaMalloc((void**)&workloadptr,  workloadsize * sizeof(int));
//    setDeviceValArr<int, int><<<1,1>>>(workloadptr, workloadsize, 0);
//    workload_split<<<1,1>>>(workloadptr, bsrRowPtr, nblockrows, bsrColInd, MIN);
////    printResVec<int><<<1,1>>>(workloadptr, workloadsize);
//
//    int gridDim_ws = (int)ceil(cbrt((double)workloadsize));
//    dim3 grid_ws(gridDim_ws, gridDim_ws, gridDim_ws);
//    printf("cbrt(workloadsize) = %d\n", gridDim_ws);

    int MAX = atoi(argv[2]);

    // count (estimate) workload
    int *workloadsizeptr;
    cudaMalloc((void**)&workloadsizeptr, 1 * sizeof(int));
    count_workload_merge_and_split<<<1,1>>>(workloadsizeptr, bsrRowPtr, nblockrows, bsrColInd, MAX);

//    int k;
//    std::cin >> k;

    int workloadsize;
    cudaMemcpy(&workloadsize, workloadsizeptr, sizeof(int) * 1, cudaMemcpyDeviceToHost);
    printf("workloadsize: %d (nblockrows: %d)\n", workloadsize, nblockrows);

    // set grid size
    int gridDim_ws = (int)ceil(cbrt((double)workloadsize));
    dim3 grid_ws(gridDim_ws, gridDim_ws, gridDim_ws);
    printf("cbrt(workloadsize) = %d\n", gridDim_ws);

    // get workload info list
    int *workload_size_list;
    cudaMalloc((void**)&workload_size_list, workloadsize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(workload_size_list, workloadsize, 0);

    int *workload_info_list;
    cudaMalloc((void**)&workload_info_list,  workloadsize * (MAX+2) * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(workload_info_list, workloadsize * (MAX+2), 0); // <-- this MAX is just a temporary number

    int *workload_info_list_size_ptr;
    cudaMalloc((void**)&workload_info_list_size_ptr, 1 * sizeof(int));
    getWorkloadInfo<<<1,1>>>(bsrRowPtr, nblockrows, MAX,
                             workload_size_list, workload_info_list, workloadsize, workload_info_list_size_ptr);

    int workload_info_list_size;
    cudaMemcpy(&workload_info_list_size, workload_info_list_size_ptr, sizeof(int) * 1, cudaMemcpyDeviceToHost);
    printf("workload_info_list_size: %d\n", workload_info_list_size);
//    printResVec<int><<<1,1>>>(workload_size_list, workloadsize);
//    printResVec<int><<<1,1>>>(workload_info_list, workload_info_list_size);
//    printWorkloadInfoList<<<1,1>>>(workload_info_list, workload_size_list, workloadsize);

    // accumulated workload_size_list to indicate start and end of workload_info_list
    int *workload_size_list_acc;
    cudaMalloc((void**)&workload_size_list_acc, (workloadsize+1) * sizeof(int));
    setWorkloadSizeListAcc<<<1,1>>>(workload_size_list_acc, workload_size_list, workloadsize);
//    printResVec<int><<<1,1>>>(workload_size_list_acc, workloadsize+1);


    //  ===== configure workload =====

    int *runtime;
    int *load;
#ifdef PROF
    cudaMalloc(&runtime, workloadsize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(runtime, workloadsize, 0);

    cudaMalloc(&load, nblockrows * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(load, workloadsize, 0);
#endif

    // ------

    GpuTimer bmv_timer;
    bmv_timer.Start();

    for (int i=0; i<TEST_TIMES; i++) { // follow warp consolidation model (32 threads per block)

        bmv64_sparse_workloadmergeNsplit<int, float><<<grid_ws, 32>>>(tA, tB, fC, bsrRowPtr, bsrColInd,
                                        workload_info_list, workload_size_list_acc,
                                        workloadsize, MAX, nblockrows, nblocks, runtime, load);
    }

    bmv_timer.Stop();
    double bmv64_time = bmv_timer.ElapsedMillis()/double(TEST_TIMES);

    // ------

#ifdef PROF
    printTimenLoadReport<<<1,1>>>(runtime, load, workloadsize); cudaFree(runtime); cudaFree(load);
#endif

    // free memory
    cudaFree(tA);
    cudaFree(tB);

    // copy result to host for verification
    float* result_bsrbmv64 = (float*)malloc(nrows * 1 * sizeof(float)); // don't care padding result
    cudaMemcpy(result_bsrbmv64, fC, nrows * 1 * sizeof(float), cudaMemcpyDeviceToHost);
//    printf("result_bsrbmv32: \n"); printResVec<float><<<1,1>>>(fC, nrows);
    printf("bsrbmv64 nnz in vec: %d\n", countNnzinVec<float>(result_bsrbmv64, nrows));

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
//    printf("csrspmvvec: \n"); printResVec<float><<<1,1>>>(dY, nrows);
    printf("cuSPARSE nnz in vec: %d\n", countNnzinVec<float>(result_cusparsecsrspmvfloat, nrows));


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
#else
    main32(argc, argv);
#endif
}
