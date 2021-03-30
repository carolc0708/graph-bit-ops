#include <iostream>
#include <sys/time.h>

#define MAX_ITER 10
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

   // get gridDim, this is to avoid nblockrows being larger than MAX_gridDim (65535?!)
    int gridDim = (int)ceil(cbrt((double)nblockrows));
    dim3 grid(gridDim, gridDim, gridDim);

    // ============================================= BSTC-32 bsr bmv
    // pagerank vector (p)
    // p fill with 1/nrows
    float* p;
    cudaMalloc((void**)&p, nblockrows * blocksize * sizeof(float));
    fillVal<<<(int)ceil(nblockrows*blocksize/1024.0), 1024>>>(p, nblockrows * blocksize, 1.f/nrows);

    // previous pagerank vector (p_prev)
    float* p_prev;
    cudaMalloc((void**)&p_prev, nblockrows * blocksize * sizeof(float));
    setDeviceValArr<int, float><<<1,1>>>(p_prev, nblockrows * blocksize , 0);

    // temporary pagerank (p_swap)
    float* p_swap;
    cudaMalloc((void**)&p_swap, nblockrows * blocksize * sizeof(float));
    setDeviceValArr<int, float><<<1,1>>>(p_swap, nblockrows * blocksize , 0);

    // residual vector (r)
    // fill r with 1.f
    float* r;
    cudaMalloc((void**)&r, nblockrows * blocksize * sizeof(float));
    fillVal<<<(int)ceil(nblockrows*blocksize/1024.0), 1024>>>(r, nblockrows * blocksize, 1.f);

    // temporary residual (r_temp)
    float* r_temp;
    cudaMalloc((void**)&r_temp, nblockrows * blocksize * sizeof(float));
    setDeviceValArr<int, float><<<1,1>>>(r_temp, nblockrows * blocksize , 0);

    float error_last = 0.f;
    float error = 1.f;
    int unvisited = nrows;

    // PageRank Parameters
    float alpha = 0.85;
    float eps   = 1e-8;

    float* errorptr;
    cudaMalloc((void**)&errorptr, sizeof(float));

    int gridDim_new = (int)ceil(cbrt((double)nblockrows/32));
    dim3 grid_new(gridDim_new, gridDim_new, gridDim_new);

    printf("nrows: %d\n", nrows);
    printf("------------------------------------\n");

    int iter;

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

    for (iter=1; error > eps && iter<=MAX_ITER; iter++) {

        //
        unvisited -= (int)(error);
        error_last = error;
        cudaMemcpy(p_prev, p, nblockrows * blocksize * sizeof(float), cudaMemcpyDeviceToDevice);

       // vxm: p = A*p + (1-alpha)*1
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
       bmv32_sparse_full<int, float><<<grid_new, 1024>>>(tA, p, p_swap, new_bsrRowPtr, new_bsrColInd, nblockrows);
       // bmvbin_timer.Stop();
       // bmvbin32_time += bmvbin_timer.ElapsedMillis();

       // solution 5
       // bmvbin_timer.Start();
       // bmv32_sparse_bin_masked_v5<int, float><<<grid_new, 1024>>>(tA, frontier1, frontier2, new_bsrRowPtr, new_bsrColInd, nblockrows, visited);
       // bmvbin_timer.Stop();
       // bmvbin32_time += bmvbin_timer.ElapsedMillis();


        // ewise add p += p_swap + (1-alpha)/nrows
        ewiseAddVal<<<(int)ceil(nblockrows*blocksize/1024.0), 1024>>>(p, nblockrows*blocksize, p_swap, (1.f-alpha)/nrows);

        // error = l2loss(p, p_prev)
        // r += p-pprev
        ewiseSubVec<<<(int)ceil(nblockrows*blocksize/1024.0), 1024>>>(r, nblockrows*blocksize, p, p_prev);

        // r_temp *= r * r
        ewiseMul<<<(int)ceil(nblockrows*blocksize/1024.0), 1024>>>(r_temp, nblockrows*blocksize, r, r);

        // reduce rtemp to error
        reduceAdd<<<(int)ceil(nblockrows*blocksize/1024.0), 1024>>>(errorptr, nblockrows*blocksize, r_temp);
        cudaMemcpy(&error, errorptr, sizeof(int), cudaMemcpyDeviceToHost);

        error = sqrt(error);
        printf("error: %d\n", error_last);
    }

    bmvbin_timer.Stop();
    bmvbin32_time = bmvbin_timer.ElapsedMillis();

    printf("------------------------------------\n");
    int niter = iter; printf("niter: %d\n", niter);
    // ------

    // free storage
    cudaFree(tA);

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
}

int main(int argc, char* argv[])
{
    main32(argc, argv);
}
