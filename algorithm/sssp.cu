#include <iostream>
#include <limits> // for std::numeric_limits<float>::max()
#include <sys/time.h>

#define TEST_TIMES 10000
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#include "mmio_highlevel.h"
#include "csr2bsr_batch_bsrbmv.cu"

/* sssp-4 */
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
    cudaMalloc((void**)&tA, ceil((float)nblocks/64) * 64 * blocksize * sizeof(uchar));

    // use batch transform as default
    csr2bsr_batch_4(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                    new_bsrRowPtr, new_bsrColInd, tA, blocksize, nblockrows, nblocks);


    free(h_cscRowInd);
    free(h_cscColPtr);

    // ============================================= input vector storage

   // get gridDim, this is to avoid nblockrows being larger than MAX_gridDim (65535?!)
    int gridDim = (int)ceil(cbrt((double)nblockrows/8));
    dim3 grid(gridDim, gridDim, gridDim);

    // ============================================= BSTC-4 bsr bmv
    // visited
    unsigned* visited;
    cudaMalloc((void**)&visited, nblockrows * blocksize * sizeof(unsigned));
    setDeviceValArr<int, unsigned><<<1,1>>>(visited, nblockrows * blocksize, 2147483647);
    setSource<<<1,1>>>(visited, 0); // source_ind = 0;

    // init frontier
    unsigned *frontier1;
    cudaMalloc((void**)&frontier1, nblockrows * blocksize * sizeof(unsigned));
    setDeviceValArr<int, unsigned><<<1,1>>>(frontier1, nblockrows*blocksize, 2147483647);
    setSource<<<1,1>>>(frontier1, 0); // source_ind = 0;

    unsigned* frontier2;
    cudaMalloc((void**)&frontier2, nblockrows * blocksize * sizeof(unsigned));
    setDeviceValArr<int, unsigned><<<1,1>>>(frontier2, nblockrows * blocksize, 77);
//    setSource<<<1,1>>>(frontier2, 0); // source_ind = 0;

    // mask vector
    unsigned* mask;
    cudaMalloc((void**)&mask, nblockrows * blocksize * sizeof(unsigned));
    setDeviceValArr<int, unsigned><<<1,1>>>(mask, nblockrows * blocksize, 0);

    //
    int f1_nvals = 1;
    unsigned succ = 1;
    unsigned *succptr;
    cudaMalloc((void**)&succptr, sizeof(unsigned));

    printf("nrows: %d\n", nrows);
    printf("------------------------------------\n");


     dim3 NT, NB;
     int nt = 1024;
     NT.x = nt;
     NT.y = 1;
     NT.z = 1;
     NB.x = (nblockrows+nt-1)/nt;
     NB.y = 1;
     NB.z = 1;


    // ------
    GpuTimer bmv_timer;
    double bmv4_time;
    bmv_timer.Start();

    int i;
    for (i=0; i<TEST_TIMES; i++) {

       // ------------------------------------------------ vxm no mask, result = min(f1 + A)
       bmv4_sparse_full_minplus<int, unsigned><<<grid, 32>>>(tA, frontier1, frontier2, new_bsrRowPtr, new_bsrColInd, nblockrows);

       // ewise add, CustomLessPlusSemiring: m = v+f2 (mask = 1 if f2 < v else 0)
       ewiseLess<<<(int)ceil(nblockrows*blocksize/1024.0), 1024>>>(mask, nblockrows*blocksize, frontier2, visited);
//       printf("mask: \n"); printResVec<<<1,1>>>(mask, nblockrows*blocksize);

       // ewise add, MinimumPlusSemiring: v = v+f2 (v = min(v, f2))
       ewiseMin<<<(int)ceil(nblockrows*blocksize/1024.0), 1024>>>(visited, nblockrows*blocksize, visited, frontier2);

       // Similar to BFS, except we need to filter out the unproductive vertices
       // here rather than as part of masked vxm
       // assign f2 with max if it is masked with 0
       assignMax<<<(int)ceil(nblockrows*blocksize/1024.0), 1024>>>(frontier2, nblockrows*blocksize, mask, 2147483647);

       // swap frontier 1 and frontier2, frontier2 fill 0
       cudaMemcpy(frontier1, frontier2, nblockrows*blocksize*sizeof(unsigned), cudaMemcpyDeviceToDevice);
//       fillValUnsigned<<<(int)ceil(nblockrows*blocksize/1024.0), 1024>>>(frontier2, nblockrows*blocksize, 0);

       // get f1 nvals

       // get succ by reduce mask
       resetSuccptrUnsigned<<<1,1>>>(succptr); //<-- use together with reduce
       reduceAddUnsigned<<<(int)ceil(nblockrows*blocksize/1024.0), 1024>>>(succptr, nblockrows*blocksize, mask);
       cudaMemcpy(&succ, succptr, sizeof(unsigned), cudaMemcpyDeviceToHost);
       printf("succ: %d\n", succ); // <-- print will slow down some time

//       int k;
//       std::cin >> k;

       // terminate condition
       if (succ == 0) break;
    }

    bmv_timer.Stop();
    bmv4_time = bmv_timer.ElapsedMillis();

    printf("------------------------------------\n");
    int niter = i; printf("niter: %d\n", niter);
    // ------


    // free storage
    cudaFree(tA);

    //============================================= check result
    printf("SSSP: %.3lf\n", bmv4_time);

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

/* sssp-32 */
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

    // ============================================= BSTC-32 bsr bmv
    // visited
    unsigned* visited;
    cudaMalloc((void**)&visited, nblockrows * blocksize * sizeof(unsigned));
    setDeviceValArr<int, unsigned><<<1,1>>>(visited, nblockrows * blocksize, 2147483647);
    setSource<<<1,1>>>(visited, 0); // source_ind = 0;

    // init frontier
    unsigned *frontier1;
    cudaMalloc((void**)&frontier1, nblockrows * blocksize * sizeof(unsigned));
    setDeviceValArr<int, unsigned><<<1,1>>>(frontier1, nblockrows*blocksize, 2147483647);
    setSource<<<1,1>>>(frontier1, 0); // source_ind = 0;

    unsigned* frontier2;
    cudaMalloc((void**)&frontier2, nblockrows * blocksize * sizeof(unsigned));
    setDeviceValArr<int, unsigned><<<1,1>>>(frontier2, nblockrows * blocksize, 77);
//    setSource<<<1,1>>>(frontier2, 0); // source_ind = 0;

    // mask vector
    unsigned* mask;
    cudaMalloc((void**)&mask, nblockrows * blocksize * sizeof(unsigned));
    setDeviceValArr<int, unsigned><<<1,1>>>(mask, nblockrows * blocksize, 0);

    //
    int f1_nvals = 1;
    unsigned succ = 1;
    unsigned *succptr;
    cudaMalloc((void**)&succptr, sizeof(unsigned));


    int gridDim_new = (int)ceil(cbrt((double)nblockrows/16));
    dim3 grid_new(gridDim_new, gridDim_new, gridDim_new);


    printf("nrows: %d\n", nrows);
    printf("------------------------------------\n");


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

    int i;
    for (i=0; i<TEST_TIMES; i++) {

       // ------------------------------------------------ vxm no mask, result = min(f1 + A)
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
//       printf("frontier1: \n"); printResVec<<<1,1>>>(frontier1, nblockrows*blocksize);

       bmv32_sparse_full_minplus<int, unsigned><<<grid_new, 512>>>(tA, frontier1, frontier2, new_bsrRowPtr, new_bsrColInd, nblockrows);
//       printf("frontier2: \n"); printResVec<<<1,1>>>(frontier2, nblockrows*blocksize);
       // bmvbin_timer.Stop();
       // bmvbin32_time += bmvbin_timer.ElapsedMillis();

       // solution 5
       // bmvbin_timer.Start();
       // bmv32_sparse_bin_masked_v5<int, ><<<grid_new, 1024>>>(tA, frontier1, frontier2, new_bsrRowPtr, new_bsrColInd, nblockrows, visited);
       // bmvbin_timer.Stop();
       // bmvbin32_time += bmvbin_timer.ElapsedMillis();
       // ------------------------------------------------

       // ewise add, CustomLessPlusSemiring: m = v+f2 (mask = 1 if f2 < v else 0)
       ewiseLess<<<(int)ceil(nblockrows*blocksize/1024.0), 1024>>>(mask, nblockrows*blocksize, frontier2, visited);
//       printf("mask: \n"); printResVec<<<1,1>>>(mask, nblockrows*blocksize);

       // ewise add, MinimumPlusSemiring: v = v+f2 (v = min(v, f2))
       ewiseMin<<<(int)ceil(nblockrows*blocksize/1024.0), 1024>>>(visited, nblockrows*blocksize, visited, frontier2);

       // Similar to BFS, except we need to filter out the unproductive vertices
       // here rather than as part of masked vxm
       // assign f2 with max if it is masked with 0
       assignMax<<<(int)ceil(nblockrows*blocksize/1024.0), 1024>>>(frontier2, nblockrows*blocksize, mask, 2147483647);

       // swap frontier 1 and frontier2, frontier2 fill 0
       cudaMemcpy(frontier1, frontier2, nblockrows*blocksize*sizeof(unsigned), cudaMemcpyDeviceToDevice);
//       fillValUnsigned<<<(int)ceil(nblockrows*blocksize/1024.0), 1024>>>(frontier2, nblockrows*blocksize, 0);

       // get f1 nvals
       
       // get succ by reduce mask
       resetSuccptrUnsigned<<<1,1>>>(succptr); //<-- use together with reduce
       reduceAddUnsigned<<<(int)ceil(nblockrows*blocksize/1024.0), 1024>>>(succptr, nblockrows*blocksize, mask);
       cudaMemcpy(&succ, succptr, sizeof(unsigned), cudaMemcpyDeviceToHost);
       printf("succ: %d\n", succ); // <-- print will slow down some time

//       int k;
//       std::cin >> k;

       // terminate condition
       if (succ == 0) break;
    }

    bmvbin_timer.Stop();
    bmvbin32_time = bmvbin_timer.ElapsedMillis();

    printf("------------------------------------\n");
    int niter = i; printf("niter: %d\n", niter);
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
#if BLOCKSIZE == 32
    main32(argc, argv);
#else
    main4(argc, argv);
#endif
}
