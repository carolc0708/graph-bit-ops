#include <iostream>
#include <sys/time.h>

#define TEST_TIMES 10000
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <vector>
#include "readMtx.hpp"

//#include "mmio_highlevel.h"
#include "csr2bsr_batch_bsrbmv.cu"

/* BFS-4 */
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

    // graphblast mmio interface =======
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<float> values;
    int nrows, ncols, nnz;
    char* dat_name;
    int directed = atoi(argv[2]);

    readMtx(filename, &row_indices, &col_indices, &values,
            &nrows, &ncols, &nnz, directed, false, &dat_name); // directed, mtxinfo

    int* h_csrRowPtr = (int*) malloc(sizeof(int) * (nrows+1));
    int* h_csrColInd = (int*) malloc(sizeof(int) * nnz);
    float* h_csrVal = (float*) malloc(sizeof(float) * nnz);

    coo2csr(h_csrRowPtr, h_csrColInd, h_csrVal,
    row_indices, col_indices, values, nrows, ncols);
    printf("nrows: %d, ncols: %d, nnz: %d\n", nrows, ncols, nnz);

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

    cudaMalloc((void**)&bsrRowPtr, sizeof(int) *(nblockrows+1));
    cusparseXcsr2bsrNnz(handle, dirA, nrows, ncols, csr_descr,
                        csrRowPtr, csrColInd, blocksize, bsr_descr, bsrRowPtr, &nblocks);
    cudaMalloc((void**)&bsrColInd, sizeof(int)*nblocks);
    printf("blocksize: %d, nblockrows: %d, nblocks: %d\n", blocksize, nblockrows, nblocks);
    unsigned bytes = (nblocks * blocksize * 1 + (nblockrows+1+nblocks) * 4);
    printf("bsr total size: "); printBytes(bytes); printf("\n");


    // csr2csc for B as A^T
    int* cscRowInd, *cscColPtr;
    float* cscVal;
    cudaMalloc(&cscRowInd, sizeof(int) * nnz);
    cudaMalloc(&cscColPtr, sizeof(int) * (nrows+1));
    cudaMalloc(&cscVal, sizeof(float) * nnz);

    cusparseHandle_t handle_csr2csc;
    cusparseCreate(&handle_csr2csc);
    cusparseScsr2csc(handle_csr2csc, nrows, ncols, nnz,
                     csrVal, csrRowPtr, csrColInd,
                     cscVal, cscRowInd, cscColPtr,
                     CUSPARSE_ACTION_NUMERIC,
                     CUSPARSE_INDEX_BASE_ZERO);
    cusparseDestroy(handle_csr2csc);

    int *h_cscRowInd, *h_cscColPtr;
    h_cscRowInd = (int*) malloc(sizeof(int) * nnz);
    h_cscColPtr = (int*) malloc(sizeof(int) * (nrows+1));
    cudaMemcpy(h_cscRowInd, cscRowInd, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cscColPtr, cscColPtr, sizeof(int) * (nrows+1), cudaMemcpyDeviceToHost);

    // csr2bsr for B & pack matrix for tB
    int* new_bsrRowPtr, *new_bsrColInd;
    cudaMalloc(&new_bsrRowPtr, sizeof(int) * (nblockrows+1));
    cudaMalloc(&new_bsrColInd, sizeof(int) * nblocks);

    // packed matrix
    uchar* tA;
    cudaMalloc((void**)&tA, nblocks * blocksize * sizeof(uchar));

    // use batch transform as default
    csr2bsr_batch_4(h_cscColPtr, h_cscRowInd, nrows, ncols, nnz,
                    new_bsrRowPtr, new_bsrColInd, tA, blocksize, nblockrows, nblocks);

    free(h_cscRowInd);
    free(h_cscColPtr);

    // ============================================= input vector storage
    // generate random vector
    srand(time(0));
	float *B = (float*)malloc((nblockrows * blocksize) * 1 * sizeof(float));
    for(int i=0 ;i<(nblockrows * blocksize); i++) B[i] = 0;
    B[0] = 1;

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
    cudaMalloc(&tB, (int)ceil((float)nblockrows/4)* 4 * sizeof(uchar));
    setDeviceValArr<int, uchar><<<1,1>>>(tB, ceil((float)nblockrows/4)*4, 0);

    // get gridDim, this is to avoid nblockrows being larger than MAX_gridDim (65535?!)
    int gridDim = (int)ceil(cbrt((double)nblockrows/4));
    dim3 grid(gridDim, gridDim, gridDim);

#ifdef VERBOSE
    printf("ceil(nblockrows/4) = %d, cbrt(nblockrows/4) = %d\n", (int)ceil((double)nblockrows/4), gridDim);
#endif

    ToBit4Row<float><<<grid, 32>>>(fB, tB, nblockrows); // dense vector


    // ============================================= BSTC-4 bsr bmv
    int nblockrows_aligned = (int)ceil((float)nblockrows/4) * 4;

    // init frontier
    uchar *frontier1;
    cudaMalloc((void**)&frontier1, nblockrows_aligned * sizeof(uchar));
    setDeviceValArr<int, uchar><<<1,1>>>(frontier1, nblockrows_aligned, 0);
    cudaMemcpy(frontier1, tB, nblockrows_aligned * sizeof(uchar), cudaMemcpyDeviceToDevice);

    uchar* frontier2;
    cudaMalloc((void**)&frontier2, nblockrows_aligned * sizeof(uchar));
    setDeviceValArr<int, uchar><<<1,1>>>(frontier2, nblockrows_aligned, 0);

    uchar* visited;
    cudaMalloc((void**)&visited, nblockrows_aligned * sizeof(uchar));

    int gridDim_2 = (int)ceil(cbrt((double)nblockrows/8));
    dim3 grid_2(gridDim_2, gridDim_2, gridDim_2);


    printf("nrows: %d\n", nrows);
    printf("------------------------------------\n");
    int succ = 0, prev_succ = 0;
    int *succptr;
    cudaMalloc((void**)&succptr, sizeof(int));
    int i;

     dim3 NT, NB;
     int nt = 1024;
     NT.x = nt;
     NT.y = 1;
     NT.z = 1;
     NB.x = (nblockrows+nt-1)/nt;
     NB.y = 1;
     NB.z = 1;

    // ------
    GpuTimer bmvbin_timer;
    double bmvbin4_time;
    bmvbin_timer.Start();

    for (i=0; i<TEST_TIMES; i++) {

       // assign new nodes to visited
       OR<uchar><<<(int)ceil(nblockrows_aligned/1024.0), 1024>>>(visited, nblockrows_aligned, frontier1);

       // frontier2 = A * frontier1, mask = visited
       // solution 1
       // bmvbin_timer.Start();
       // bmv32_sparse_bin_masked_v1<int, float><<<NB, NT>>>(tA, frontier1, frontier2, new_bsrRowPtr, new_bsrColInd, nblockrows, visited);
       // bmvbin_timer.Stop();
       // bmvbin32_time += bmvbin_timer.ElapsedMillis();

       // solution 2
       // bmvbin_timer.Start();
       // bmv32_sparse_bin_masked_v2<int, float><<<grid, 32>>>(tA, frontier1, frontier2, new_bsrRowPtr, new_bsrColInd, nblockrows, visited);
       // // Mask<<<1,1>>>(frontier2, nblockrows, visited); <-- required only when masked is not pass in
       // bmvbin_timer.Stop();
       // bmvbin32_time += bmvbin_timer.ElapsedMillis();


       // solution 3
       // bmvbin_timer.Start();
       // bmv32_sparse_bin_masked_v3<int, float><<<grid, 32>>>(tA, frontier1, frontier2, new_bsrRowPtr, new_bsrColInd, nblockrows, visited);
       // bmvbin_timer.Stop();
       // bmvbin32_time += bmvbin_timer.ElapsedMillis();

       // solution 4
       // bmvbin_timer.Start();
       bmv4_sparse_bin_masked_v4<int, uchar><<<grid_2, 32>>>(tA, frontier1, frontier2, new_bsrRowPtr, new_bsrColInd, nblockrows, visited);
       // bmvbin_timer.Stop();
       // bmvbin32_time += bmvbin_timer.ElapsedMillis();

       // solution 5
       // bmvbin_timer.Start();
       // bmv32_sparse_bin_masked_v5<int, float><<<grid_new, 1024>>>(tA, frontier1, frontier2, new_bsrRowPtr, new_bsrColInd, nblockrows, visited);
       // bmvbin_timer.Stop();
       // bmvbin32_time += bmvbin_timer.ElapsedMillis();

       // swap frontier 1 and frontier2, frontier2 fill 0
       cudaMemcpy(frontier1, frontier2, nblockrows_aligned * sizeof(uchar), cudaMemcpyDeviceToDevice);
       fillZero<uchar><<<(int)ceil(nblockrows_aligned/1024.0), 1024>>>(frontier2, nblockrows_aligned);
//       printf("result_bsrbmv4-bin: \n"); printBin8Vec<<<1,1>>>(frontier1, nblockrows_aligned);


       // get succ by reduce frontier1
       resetSuccptr<<<1,1>>>(succptr); //<-- use together with reduce
       reduce<uchar><<<(int)ceil(nblockrows_aligned/1024.0), 1024>>>(frontier1, nblockrows_aligned, succptr);
       // reduce_naive<<<1,1>>>(frontier1, nblockrows, succptr);
       cudaMemcpy(&succ, succptr, sizeof(int), cudaMemcpyDeviceToHost);
       printf("succ: %d\n", succ); // <-- print will slow down some time

//       int k;
//       std::cin >> k;

       // terminate condition
       if (succ == 0) break;
    }

    bmvbin_timer.Stop();
    bmvbin4_time = bmvbin_timer.ElapsedMillis();

    printf("------------------------------------\n");
    int niter = i+1; printf("niter: %d\n", niter);
    // ------


    // free storage
    cudaFree(tA);
    cudaFree(tB);

#ifdef VERBOSE
//    printf("result_bsrbmv32: \n"); printResVec<float><<<1,1>>>(fC, nrows);
//    printf("result_bsrbmv32-bin: \n"); printBin32Vec<<<1,1>>>(tC, nblockrows);
//    verify32BinResVec<<<1,1>>>(tC, fC, nblockrows);
//    printf("bsrbmv32 nnz in vec: %d\n", countNnzinVec<float>(result_bsrbmv32, nrows));
#endif

    //============================================= check result
    printf("BSR BMV-4-bin: %.3lf\n", bmvbin4_time);

    //============================================= free memory
    // free cusparse bsr metadata
    cusparseDestroyMatDescr(csr_descr);
    cusparseDestroyMatDescr(bsr_descr);
    cusparseDestroy(handle);

    // free mem
    free(h_csrRowPtr);
    free(h_csrColInd);
    free(h_csrVal);

    cudaFree(csrRowPtr);
    cudaFree(csrColInd);
    cudaFree(csrVal);

    cudaFree(fB);
}

/* BFS-32 */
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

    // graphblast mmio interface =======
    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<float> values;
    int nrows, ncols, nnz;
    char* dat_name;
    int directed = atoi(argv[2]);

    readMtx(filename, &row_indices, &col_indices, &values,
            &nrows, &ncols, &nnz, directed, false, &dat_name); // directed, mtxinfo

    int* h_csrRowPtr = (int*) malloc(sizeof(int) * (nrows+1));
    int* h_csrColInd = (int*) malloc(sizeof(int) * nnz);
    float* h_csrVal = (float*) malloc(sizeof(float) * nnz);

    coo2csr(h_csrRowPtr, h_csrColInd, h_csrVal,
    row_indices, col_indices, values, nrows, ncols);
    printf("nrows: %d, ncols: %d, nnz: %d\n", nrows, ncols, nnz);

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


    // csr2csc for B as A^T
    int* cscRowInd, *cscColPtr;
    float* cscVal;
    cudaMalloc(&cscRowInd, sizeof(int) * nnz);
    cudaMalloc(&cscColPtr, sizeof(int) * (nrows+1));
    cudaMalloc(&cscVal, sizeof(float) * nnz);

    cusparseHandle_t handle_csr2csc;
    cusparseCreate(&handle_csr2csc);
    cusparseScsr2csc(handle_csr2csc, nrows, ncols, nnz,
                     csrVal, csrRowPtr, csrColInd,
                     cscVal, cscRowInd, cscColPtr,
                     CUSPARSE_ACTION_NUMERIC,
                     CUSPARSE_INDEX_BASE_ZERO);
    cusparseDestroy(handle_csr2csc);

    int *h_cscRowInd, *h_cscColPtr;
    h_cscRowInd = (int*) malloc(sizeof(int) * nnz);
    h_cscColPtr = (int*) malloc(sizeof(int) * (nrows+1));
    cudaMemcpy(h_cscRowInd, cscRowInd, sizeof(int) * nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cscColPtr, cscColPtr, sizeof(int) * (nrows+1), cudaMemcpyDeviceToHost);

    // csr2bsr for B & pack matrix for tB
    int* new_bsrRowPtr, *new_bsrColInd;
    cudaMalloc(&new_bsrRowPtr, sizeof(int) * (nblockrows+1));
    cudaMalloc(&new_bsrColInd, sizeof(int) * nblocks);

    // packed matrix
    unsigned* tA;
    cudaMalloc((void**)&tA, nblocks * blocksize * sizeof(unsigned));

    // use batch transform as default
    csr2bsr_batch_32(h_cscColPtr, h_cscRowInd, nrows, ncols, nnz,
                     new_bsrRowPtr, new_bsrColInd, tA, blocksize, nblockrows, nblocks);

    free(h_cscRowInd);
    free(h_cscColPtr);

    // ============================================= input vector storage
    // generate random vector
    srand(time(0));
	float *B = (float*)malloc((nblockrows * blocksize) * 1 * sizeof(float));
//	for (int i = 0; i < (nblockrows * blocksize) * 1; i ++)
//    {
//        float x = (float)rand() / RAND_MAX;
//        if (i >= ncols) B[i] = 0;
//        else B[i] = (x > 0.5) ? 1 : 0;
//    }
    for(int i=0 ;i<(nblockrows * blocksize); i++) B[i] = 0;
    B[0] = 1;

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

    // init frontier
    unsigned *frontier1;
    cudaMalloc((void**)&frontier1, nblockrows * sizeof(unsigned));
    setDeviceValArr<int, unsigned><<<1,1>>>(frontier1, nblockrows, 0);
    cudaMemcpy(frontier1, tB, nblockrows * sizeof(unsigned), cudaMemcpyDeviceToDevice);

    unsigned* frontier2;
    cudaMalloc((void**)&frontier2, nblockrows * sizeof(unsigned));
    setDeviceValArr<int, unsigned><<<1,1>>>(frontier2, nblockrows, 0);

    unsigned* visited;
    cudaMalloc((void**)&visited, nblockrows * sizeof(unsigned));
    

    int gridDim_new = (int)ceil(cbrt((double)nblockrows/32));
    dim3 grid_new(gridDim_new, gridDim_new, gridDim_new);


    printf("nrows: %d\n", nrows);
    printf("------------------------------------\n");
    int succ = 0, prev_succ = 0;
    int *succptr;
    cudaMalloc((void**)&succptr, sizeof(int));
    int i;

     dim3 NT, NB;
     int nt = 1024;
     NT.x = nt;
     NT.y = 1;
     NT.z = 1;
     NB.x = (nblockrows+nt-1)/nt;
     NB.y = 1;
     NB.z = 1;

    // ------
    GpuTimer bmvbin_timer;
    double bmvbin32_time;
    bmvbin_timer.Start();

    for (i=0; i<TEST_TIMES; i++) {

       // assign new nodes to visited
       OR<unsigned><<<(int)ceil(nblockrows/1024.0), 1024>>>(visited, nblockrows, frontier1);

       // frontier2 = A * frontier1, mask = visited
       // solution 1
       // bmvbin_timer.Start();
       // bmv32_sparse_bin_masked_v1<int, float><<<NB, NT>>>(tA, frontier1, frontier2, new_bsrRowPtr, new_bsrColInd, nblockrows, visited);
       // bmvbin_timer.Stop();
       // bmvbin32_time += bmvbin_timer.ElapsedMillis();

       // solution 2
       // bmvbin_timer.Start();
       // bmv32_sparse_bin_masked_v2<int, float><<<grid, 32>>>(tA, frontier1, frontier2, new_bsrRowPtr, new_bsrColInd, nblockrows, visited);
       // // Mask<<<1,1>>>(frontier2, nblockrows, visited); <-- required only when masked is not pass in
       // bmvbin_timer.Stop();
       // bmvbin32_time += bmvbin_timer.ElapsedMillis();


       // solution 3
       // bmvbin_timer.Start();
       // bmv32_sparse_bin_masked_v3<int, float><<<grid, 32>>>(tA, frontier1, frontier2, new_bsrRowPtr, new_bsrColInd, nblockrows, visited);
       // bmvbin_timer.Stop();
       // bmvbin32_time += bmvbin_timer.ElapsedMillis();

       // solution 4
       // bmvbin_timer.Start();
       bmv32_sparse_bin_masked_v4<int, unsigned><<<grid_new, 1024>>>(tA, frontier1, frontier2, new_bsrRowPtr, new_bsrColInd, nblockrows, visited);
       // bmvbin_timer.Stop();
       // bmvbin32_time += bmvbin_timer.ElapsedMillis();

       // solution 5
       // bmvbin_timer.Start();
       // bmv32_sparse_bin_masked_v5<int, float><<<grid_new, 1024>>>(tA, frontier1, frontier2, new_bsrRowPtr, new_bsrColInd, nblockrows, visited);
       // bmvbin_timer.Stop();
       // bmvbin32_time += bmvbin_timer.ElapsedMillis();

       // swap frontier 1 and frontier2, frontier2 fill 0
       cudaMemcpy(frontier1, frontier2, nblockrows * sizeof(unsigned), cudaMemcpyDeviceToDevice);
       fillZero<unsigned><<<(int)ceil(nblockrows/1024.0), 1024>>>(frontier2, nblockrows);
//       printf("result_bsrbmv32-bin: \n"); printBin32Vec<<<1,1>>>(frontier1, nblockrows);
       
       
       // get succ by reduce frontier1
       resetSuccptr<<<1,1>>>(succptr); //<-- use together with reduce
       reduce<unsigned><<<(int)ceil(nblockrows/1024.0), 1024>>>(frontier1, nblockrows, succptr);
       // reduce_naive<<<1,1>>>(frontier1, nblockrows, succptr);
       cudaMemcpy(&succ, succptr, sizeof(int), cudaMemcpyDeviceToHost);
       printf("succ: %d\n", succ); // <-- print will slow down some time

//       int k;
//       std::cin >> k;
       
       // terminate condition
       if (succ == 0) break;
    }

    bmvbin_timer.Stop();
    bmvbin32_time = bmvbin_timer.ElapsedMillis();

    printf("------------------------------------\n");
    int niter = i+1; printf("niter: %d\n", niter);
    // ------


    // free storage
    cudaFree(tA);
    cudaFree(tB);

#ifdef VERBOSE
//    printf("result_bsrbmv32: \n"); printResVec<float><<<1,1>>>(fC, nrows);
//    printf("result_bsrbmv32-bin: \n"); printBin32Vec<<<1,1>>>(tC, nblockrows);
//    verify32BinResVec<<<1,1>>>(tC, fC, nblockrows);
//    printf("bsrbmv32 nnz in vec: %d\n", countNnzinVec<float>(result_bsrbmv32, nrows));
#endif

    //============================================= check result
    printf("BSR BMV-32-bin: %.3lf\n", bmvbin32_time);

    //============================================= free memory
    // free cusparse bsr metadata
    cusparseDestroyMatDescr(csr_descr);
    cusparseDestroyMatDescr(bsr_descr);
    cusparseDestroy(handle);

    // free mem
    free(h_csrRowPtr);
    free(h_csrColInd);
    free(h_csrVal);

    cudaFree(csrRowPtr);
    cudaFree(csrColInd);
    cudaFree(csrVal);

    cudaFree(fB);
}

int main(int argc, char* argv[])
{
#if BLOCKSIZE == 32
    main32(argc, argv);
#else
    main4(argc, argv);
#endif
}
