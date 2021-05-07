#include <iostream>
#include <sys/time.h>

#define MAX_ITER 10000
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#include <vector>
#include "readMtx.hpp"

//#include "mmio_highlevel.h"
#include "csr2bsr_batch_bsrbmv.cu"

#include <unordered_set>

// functions from graphblast =======
template <typename Index>
int SimpleVerifyCc(Index                   nrows,
                   const Index*            h_csrRowPtr,
                   const Index*            h_csrColInd,
                   const std::vector<int>& h_cc_cpu,
                   bool                    suppress_zero)
{
  int num_error = 0;
  std::unordered_set<int> dict;

  for (Index row = 0; row < nrows; ++row) {
    int row_label = h_cc_cpu[row];
    if (dict.find(row_label) == dict.end())
      dict.insert(row_label);

    if (row_label == 0 && num_error == 0 && !suppress_zero)
      std::cout << "\nINCORRECT: [" << row << "]: has no component.\n";

    Index row_start = h_csrRowPtr[row];
    Index row_end   = h_csrRowPtr[row+1];
    for (; row_start < row_end; ++row_start) {
      Index col = h_csrColInd[row_start];
      int col_label = h_cc_cpu[col];
      if (col_label != row_label) {
        if (num_error == 0) {
          std::cout << "\nINCORRECT: [" << row << "]: ";
          std::cout << row_label << " != " << col_label << " [" << col <<
            "]\n";
        }
        num_error++;
      }
    }
  }
  if (num_error == 0)
    std::cout << "\nCORRECT\n";
  else
    std::cout << num_error << " errors occurred.\n";
  std::cout << "Connected components found with " << dict.size();
  std::cout << " components.\n";
}

// functions from graphblast =======

/* cc-4 */
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

    // packed matrix
    uchar* tA;
    cudaMalloc((void**)&tA, nblocks * blocksize * sizeof(uchar));

    // use batch transform as default
    csr2bsr_batch_4(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                     bsrRowPtr, bsrColInd, tA, blocksize, nblockrows, nblocks);

//    free(h_cscRowInd);
//    free(h_cscColPtr);

    // ============================================= BSTC-4 bsr bmv
    // Difference vector
    int* diff;
    cudaMalloc((void**)&diff, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(diff, nblockrows * blocksize , 0);

    // Parent vector
    int* parent;
    cudaMalloc((void**)&parent, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(parent, nblockrows * blocksize , 0);
    int* parent_temp;
    cudaMalloc((void**)&parent_temp, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(parent_temp, nblockrows * blocksize , 0);

    // grandparent
    int* grandparent;
    cudaMalloc((void**)&grandparent, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(grandparent, nblockrows * blocksize , 0);
    int* grandparent_temp;
    cudaMalloc((void**)&grandparent_temp, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(grandparent_temp, nblockrows * blocksize , 0);

    // min neighbor grandparent vector
    int* min_neighbor_parent;
    cudaMalloc((void**)&min_neighbor_parent, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(min_neighbor_parent, nblockrows * blocksize , 0);
    int* min_neighbor_parent_temp;
    cudaMalloc((void**)&min_neighbor_parent_temp, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(min_neighbor_parent_temp, nblockrows * blocksize , 0);

    // fill ascending
    // parent, min_neighbor_parent, min_neighbor_parent_temp, grandparent, grandparent_temp
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(parent, nrows);
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(min_neighbor_parent, nrows);
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(min_neighbor_parent_temp, nrows);
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent, nrows);
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent_temp, nrows);

    int gridDim_new = (int)ceil(cbrt((double)nblockrows/32));
    dim3 grid_new(gridDim_new, gridDim_new, gridDim_new);

    int gridDim = (int)ceil(cbrt((double)nblockrows));
    dim3 grid(gridDim, gridDim, gridDim);

    int succ = 0;
    int *succptr;
    cudaMalloc((void**)&succptr, sizeof(int));

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
    double bmvbin4_time;
    bmvbin_timer.Start();

    for (iter=1; iter<=MAX_ITER; iter++) {

        // parent_temp = parent
        initVec<<<(int)ceil(nrows/1024.0), 1024>>>(parent_temp, nrows);
        cudaMemcpy(parent_temp, parent, nrows * sizeof(int), cudaMemcpyDeviceToDevice);

        // 1) Stochastic hooking.
        // mngf[u] = A x gf
        // mxv
        bmv4_sparse_full_cc_new_2<int, int><<<grid_new, 1024>>>(tA, grandparent, min_neighbor_parent_temp, bsrRowPtr, bsrColInd, nblockrows);
//        printResVec<int><<<1,1>>>(min_neighbor_parent_temp, nrows);
        ewiseMin<<<(int)ceil(nrows/1024.0), 1024>>>(min_neighbor_parent, nrows, min_neighbor_parent, min_neighbor_parent_temp);
//        printResVec<int><<<1,1>>>(min_neighbor_parent, nrows);

        // f[f[u]] = mngf[u]
        // assignscatter /*TODO: change to gblast kernel */
        assignScatter<<<(int)ceil(nrows/1024.0), 1024>>>(parent, nrows, parent_temp, min_neighbor_parent);
//        printResVec<int><<<1,1>>>(parent, nrows);

        // 2) Aggressive hooking.
        // f = min(f, mngf)
        ewiseMin<<<(int)ceil(nrows/1024.0), 1024>>>(parent, nrows, parent, min_neighbor_parent);
//        printResVec<int><<<1,1>>>(parent, nrows);

        // 3) Shortcutting.
        // f = min(f, gf)
        ewiseMin<<<(int)ceil(nrows/1024.0), 1024>>>(parent, nrows, parent, parent_temp);
//        printResVec<int><<<1,1>>>(parent, nrows);

        // 4) Calculate grandparents.
        // gf[u] = f[f[u]]
        // extractgather /* TODO: change to gblast kernel */
        extractGather<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent, nrows, parent, parent);
//        printResVec<int><<<1,1>>>(grandparent, nrows);

        // 5) Check termination
        ewiseNotEqual<<<(int)ceil(nrows/1024.0), 1024>>>(diff, nrows, grandparent_temp, grandparent);
//        printResVec<int><<<1,1>>>(diff, nrows);

        // reduce
        resetSuccptr<<<1,1>>>(succptr);
        reduceAddInt<<<(int)ceil(nrows/1024.0), 1024>>>(succptr, nrows, diff);
        cudaMemcpy(&succ, succptr, sizeof(int), cudaMemcpyDeviceToHost);

        if (succ == 0) break;

        // grandparent_temp = grandparent
        initVec<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent_temp, nrows);
        cudaMemcpy(grandparent_temp, grandparent, nrows * sizeof(int), cudaMemcpyDeviceToDevice);

        // 6) Similar to BFS and SSSP, we should filter out the unproductive
        // vertices from the next iteration.
        // assign
//        printResVec<int><<<1,1>>>(grandparent, nrows);
        assignMax<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent, nrows, diff, 2147483647);
//        printResVec<int><<<1,1>>>(grandparent, nrows);

        // copy result to output
        // v = parent, v is a vector

        printf("succ: %d\n", succ);

//        int k;
//        std::cin >> k;
    }

    bmvbin_timer.Stop();
    bmvbin4_time = bmvbin_timer.ElapsedMillis();

    printf("------------------------------------\n");
    int niter = iter; printf("niter: %d\n", niter);
    // ------

    // result verification /* TODO: change to gblast verify function */
    if (niter > 1) {
        int* h_parent = (int*) malloc(sizeof(int) * nrows);
        cudaMemcpy(h_parent, parent, nrows * sizeof(int), cudaMemcpyDeviceToHost);
        std::vector<int> h_cc_gpu(h_parent, h_parent + nrows);
        SimpleVerifyCc(nrows, h_csrRowPtr, h_csrColInd, h_cc_gpu, true/*suppres_zero*/);
    }

    // free storage
    cudaFree(tA);

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
}

/* cc-8 */
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
    cudaMalloc((void**)&tA, nblocks * blocksize * sizeof(uchar));

    // use batch transform as default
    csr2bsr_batch_8(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                    bsrRowPtr, bsrColInd, tA, blocksize, nblockrows, nblocks);

//    free(h_cscRowInd);
//    free(h_cscColPtr);

    // ============================================= input vector storage

   // get gridDim, this is to avoid nblockrows being larger than MAX_gridDim (65535?!)
    int gridDim = (int)ceil(cbrt((double)nblockrows));
    dim3 grid(gridDim, gridDim, gridDim);

    // ============================================= BSTC-16 bsr bmv
    // Difference vector
    int* diff;
    cudaMalloc((void**)&diff, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(diff, nblockrows * blocksize , 0);

    // Parent vector
    int* parent;
    cudaMalloc((void**)&parent, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(parent, nblockrows * blocksize , 0);
    int* parent_temp;
    cudaMalloc((void**)&parent_temp, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(parent_temp, nblockrows * blocksize , 0);

    // grandparent
    int* grandparent;
    cudaMalloc((void**)&grandparent, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(grandparent, nblockrows * blocksize , 0);
    int* grandparent_temp;
    cudaMalloc((void**)&grandparent_temp, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(grandparent_temp, nblockrows * blocksize , 0);

    // min neighbor grandparent vector
    int* min_neighbor_parent;
    cudaMalloc((void**)&min_neighbor_parent, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(min_neighbor_parent, nblockrows * blocksize , 0);
    int* min_neighbor_parent_temp;
    cudaMalloc((void**)&min_neighbor_parent_temp, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(min_neighbor_parent_temp, nblockrows * blocksize , 0);

    // fill ascending
    // parent, min_neighbor_parent, min_neighbor_parent_temp, grandparent, grandparent_temp
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(parent, nrows);
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(min_neighbor_parent, nrows);
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(min_neighbor_parent_temp, nrows);
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent, nrows);
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent_temp, nrows);

    int gridDim_new = (int)ceil(cbrt((double)nblockrows/32));
    dim3 grid_new(gridDim_new, gridDim_new, gridDim_new);

    int succ = 0;
    int *succptr;
    cudaMalloc((void**)&succptr, sizeof(int));

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
    double bmvbin8_time;
    bmvbin_timer.Start();

    for (iter=1; iter<=MAX_ITER; iter++) {

        // parent_temp = parent
        initVec<<<(int)ceil(nrows/1024.0), 1024>>>(parent_temp, nrows);
        cudaMemcpy(parent_temp, parent, nrows * sizeof(int), cudaMemcpyDeviceToDevice);

        // 1) Stochastic hooking.
        // mngf[u] = A x gf
        // mxv
        bmv8_sparse_full_cc_new_2<int, int><<<grid_new, 1024>>>(tA, grandparent, min_neighbor_parent_temp, bsrRowPtr, bsrColInd, nblockrows);
//        printResVec<int><<<1,1>>>(min_neighbor_parent_temp, nrows);
        ewiseMin<<<(int)ceil(nrows/1024.0), 1024>>>(min_neighbor_parent, nrows, min_neighbor_parent, min_neighbor_parent_temp);
//        printResVec<int><<<1,1>>>(min_neighbor_parent, nrows);

        // f[f[u]] = mngf[u]
        // assignscatter /*TODO: change to gblast kernel */
        assignScatter<<<(int)ceil(nrows/1024.0), 1024>>>(parent, nrows, parent_temp, min_neighbor_parent);
//        printResVec<int><<<1,1>>>(parent, nrows);

        // 2) Aggressive hooking.
        // f = min(f, mngf)
        ewiseMin<<<(int)ceil(nrows/1024.0), 1024>>>(parent, nrows, parent, min_neighbor_parent);
//        printResVec<int><<<1,1>>>(parent, nrows);

        // 3) Shortcutting.
        // f = min(f, gf)
        ewiseMin<<<(int)ceil(nrows/1024.0), 1024>>>(parent, nrows, parent, parent_temp);
//        printResVec<int><<<1,1>>>(parent, nrows);

        // 4) Calculate grandparents.
        // gf[u] = f[f[u]]
        // extractgather /* TODO: change to gblast kernel */
        extractGather<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent, nrows, parent, parent);
//        printResVec<int><<<1,1>>>(grandparent, nrows);

        // 5) Check termination
        ewiseNotEqual<<<(int)ceil(nrows/1024.0), 1024>>>(diff, nrows, grandparent_temp, grandparent);
//        printResVec<int><<<1,1>>>(diff, nrows);

        // reduce
        resetSuccptr<<<1,1>>>(succptr);
        reduceAddInt<<<(int)ceil(nrows/1024.0), 1024>>>(succptr, nrows, diff);
        cudaMemcpy(&succ, succptr, sizeof(int), cudaMemcpyDeviceToHost);

        if (succ == 0) break;

        // grandparent_temp = grandparent
        initVec<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent_temp, nrows);
        cudaMemcpy(grandparent_temp, grandparent, nrows * sizeof(int), cudaMemcpyDeviceToDevice);

        // 6) Similar to BFS and SSSP, we should filter out the unproductive
        // vertices from the next iteration.
        // assign
//        printResVec<int><<<1,1>>>(grandparent, nrows);
        assignMax<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent, nrows, diff, 2147483647);
//        printResVec<int><<<1,1>>>(grandparent, nrows);

        // copy result to output
        // v = parent, v is a vector

        printf("succ: %d\n", succ);

//        int k;
//        std::cin >> k;
    }

    bmvbin_timer.Stop();
    bmvbin8_time = bmvbin_timer.ElapsedMillis();

    printf("------------------------------------\n");
    int niter = iter; printf("niter: %d\n", niter);
    // ------

    // result verification /* TODO: change to gblast verify function */
    if (niter > 1) {
        int* h_parent = (int*) malloc(sizeof(int) * nrows);
        cudaMemcpy(h_parent, parent, nrows * sizeof(int), cudaMemcpyDeviceToHost);
        std::vector<int> h_cc_gpu(h_parent, h_parent + nrows);
        SimpleVerifyCc(nrows, h_csrRowPtr, h_csrColInd, h_cc_gpu, true/*suppres_zero*/);
    }

    // free storage
    cudaFree(tA);

    //============================================= check result
    printf("BSR BMV-8-bin: %.3lf\n", bmvbin8_time);

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

/* cc-16 */
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
    cudaMalloc((void**)&tA, nblocks * blocksize * sizeof(ushort));

    // use batch transform as default
    csr2bsr_batch_16(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                     bsrRowPtr, bsrColInd, tA, blocksize, nblockrows, nblocks);

//    free(h_cscRowInd);
//    free(h_cscColPtr);

    // ============================================= input vector storage

   // get gridDim, this is to avoid nblockrows being larger than MAX_gridDim (65535?!)
    int gridDim = (int)ceil(cbrt((double)nblockrows));
    dim3 grid(gridDim, gridDim, gridDim);

    // ============================================= BSTC-16 bsr bmv
    // Difference vector
    int* diff;
    cudaMalloc((void**)&diff, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(diff, nblockrows * blocksize , 0);

    // Parent vector
    int* parent;
    cudaMalloc((void**)&parent, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(parent, nblockrows * blocksize , 0);
    int* parent_temp;
    cudaMalloc((void**)&parent_temp, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(parent_temp, nblockrows * blocksize , 0);

    // grandparent
    int* grandparent;
    cudaMalloc((void**)&grandparent, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(grandparent, nblockrows * blocksize , 0);
    int* grandparent_temp;
    cudaMalloc((void**)&grandparent_temp, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(grandparent_temp, nblockrows * blocksize , 0);

    // min neighbor grandparent vector
    int* min_neighbor_parent;
    cudaMalloc((void**)&min_neighbor_parent, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(min_neighbor_parent, nblockrows * blocksize , 0);
    int* min_neighbor_parent_temp;
    cudaMalloc((void**)&min_neighbor_parent_temp, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(min_neighbor_parent_temp, nblockrows * blocksize , 0);

    // fill ascending
    // parent, min_neighbor_parent, min_neighbor_parent_temp, grandparent, grandparent_temp
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(parent, nrows);
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(min_neighbor_parent, nrows);
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(min_neighbor_parent_temp, nrows);
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent, nrows);
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent_temp, nrows);

    int gridDim_new = (int)ceil(cbrt((double)nblockrows/32));
    dim3 grid_new(gridDim_new, gridDim_new, gridDim_new);

    int succ = 0;
    int *succptr;
    cudaMalloc((void**)&succptr, sizeof(int));

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
    double bmvbin16_time;
    bmvbin_timer.Start();

    for (iter=1; iter<=MAX_ITER; iter++) {

        // parent_temp = parent
        initVec<<<(int)ceil(nrows/1024.0), 1024>>>(parent_temp, nrows);
        cudaMemcpy(parent_temp, parent, nrows * sizeof(int), cudaMemcpyDeviceToDevice);

        // 1) Stochastic hooking.
        // mngf[u] = A x gf
        // mxv
        bmv16_sparse_full_cc_new_2<int, int><<<grid_new, 1024>>>(tA, grandparent, min_neighbor_parent_temp, bsrRowPtr, bsrColInd, nblockrows);
//        printResVec<int><<<1,1>>>(min_neighbor_parent_temp, nrows);
        ewiseMin<<<(int)ceil(nrows/1024.0), 1024>>>(min_neighbor_parent, nrows, min_neighbor_parent, min_neighbor_parent_temp);
//        printResVec<int><<<1,1>>>(min_neighbor_parent, nrows);

        // f[f[u]] = mngf[u]
        // assignscatter /*TODO: change to gblast kernel */
        assignScatter<<<(int)ceil(nrows/1024.0), 1024>>>(parent, nrows, parent_temp, min_neighbor_parent);
//        printResVec<int><<<1,1>>>(parent, nrows);

        // 2) Aggressive hooking.
        // f = min(f, mngf)
        ewiseMin<<<(int)ceil(nrows/1024.0), 1024>>>(parent, nrows, parent, min_neighbor_parent);
//        printResVec<int><<<1,1>>>(parent, nrows);

        // 3) Shortcutting.
        // f = min(f, gf)
        ewiseMin<<<(int)ceil(nrows/1024.0), 1024>>>(parent, nrows, parent, parent_temp);
//        printResVec<int><<<1,1>>>(parent, nrows);

        // 4) Calculate grandparents.
        // gf[u] = f[f[u]]
        // extractgather /* TODO: change to gblast kernel */
        extractGather<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent, nrows, parent, parent);
//        printResVec<int><<<1,1>>>(grandparent, nrows);

        // 5) Check termination
        ewiseNotEqual<<<(int)ceil(nrows/1024.0), 1024>>>(diff, nrows, grandparent_temp, grandparent);
//        printResVec<int><<<1,1>>>(diff, nrows);

        // reduce
        resetSuccptr<<<1,1>>>(succptr);
        reduceAddInt<<<(int)ceil(nrows/1024.0), 1024>>>(succptr, nrows, diff);
        cudaMemcpy(&succ, succptr, sizeof(int), cudaMemcpyDeviceToHost);

        if (succ == 0) break;

        // grandparent_temp = grandparent
        initVec<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent_temp, nrows);
        cudaMemcpy(grandparent_temp, grandparent, nrows * sizeof(int), cudaMemcpyDeviceToDevice);

        // 6) Similar to BFS and SSSP, we should filter out the unproductive
        // vertices from the next iteration.
        // assign
//        printResVec<int><<<1,1>>>(grandparent, nrows);
        assignMax<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent, nrows, diff, 2147483647);
//        printResVec<int><<<1,1>>>(grandparent, nrows);

        // copy result to output
        // v = parent, v is a vector

        printf("succ: %d\n", succ);

//        int k;
//        std::cin >> k;
    }

    bmvbin_timer.Stop();
    bmvbin16_time = bmvbin_timer.ElapsedMillis();

    printf("------------------------------------\n");
    int niter = iter; printf("niter: %d\n", niter);
    // ------

    // result verification /* TODO: change to gblast verify function */
    if (niter > 1) {
        int* h_parent = (int*) malloc(sizeof(int) * nrows);
        cudaMemcpy(h_parent, parent, nrows * sizeof(int), cudaMemcpyDeviceToHost);
        std::vector<int> h_cc_gpu(h_parent, h_parent + nrows);
        SimpleVerifyCc(nrows, h_csrRowPtr, h_csrColInd, h_cc_gpu, true/*suppres_zero*/);
    }

    // free storage
    cudaFree(tA);

    //============================================= check result
    printf("BSR BMV-16-bin: %.3lf\n", bmvbin16_time);

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

/* cc-32 */
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

    // packed matrix
    unsigned* tA;
    cudaMalloc((void**)&tA, nblocks * blocksize * sizeof(unsigned));

    // use batch transform as default
    csr2bsr_batch_32(h_csrRowPtr, h_csrColInd, nrows, ncols, nnz,
                     bsrRowPtr, bsrColInd, tA, blocksize, nblockrows, nblocks);

//    free(h_cscRowInd);
//    free(h_cscColPtr);

    // ============================================= input vector storage

   // get gridDim, this is to avoid nblockrows being larger than MAX_gridDim (65535?!)
    int gridDim = (int)ceil(cbrt((double)nblockrows));
    dim3 grid(gridDim, gridDim, gridDim);

    // ============================================= BSTC-32 bsr bmv
    // Difference vector
    int* diff;
    cudaMalloc((void**)&diff, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(diff, nblockrows * blocksize , 0);

    // Parent vector
    int* parent;
    cudaMalloc((void**)&parent, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(parent, nblockrows * blocksize , 0);
    int* parent_temp;
    cudaMalloc((void**)&parent_temp, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(parent_temp, nblockrows * blocksize , 0);

    // grandparent
    int* grandparent;
    cudaMalloc((void**)&grandparent, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(grandparent, nblockrows * blocksize , 0);
    int* grandparent_temp;
    cudaMalloc((void**)&grandparent_temp, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(grandparent_temp, nblockrows * blocksize , 0);

    // min neighbor grandparent vector
    int* min_neighbor_parent;
    cudaMalloc((void**)&min_neighbor_parent, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(min_neighbor_parent, nblockrows * blocksize , 0);
    int* min_neighbor_parent_temp;
    cudaMalloc((void**)&min_neighbor_parent_temp, nblockrows * blocksize * sizeof(int));
    setDeviceValArr<int, int><<<1,1>>>(min_neighbor_parent_temp, nblockrows * blocksize , 0);

    // fill ascending
    // parent, min_neighbor_parent, min_neighbor_parent_temp, grandparent, grandparent_temp
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(parent, nrows);
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(min_neighbor_parent, nrows);
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(min_neighbor_parent_temp, nrows);
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent, nrows);
    fillAscending<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent_temp, nrows);

    int gridDim_new = (int)ceil(cbrt((double)nblockrows/32));
    dim3 grid_new(gridDim_new, gridDim_new, gridDim_new);

    int succ = 0;
    int *succptr;
    cudaMalloc((void**)&succptr, sizeof(int));

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

    for (iter=1; iter<=MAX_ITER; iter++) {

        // parent_temp = parent
        initVec<<<(int)ceil(nrows/1024.0), 1024>>>(parent_temp, nrows);
        cudaMemcpy(parent_temp, parent, nrows * sizeof(int), cudaMemcpyDeviceToDevice);

        // 1) Stochastic hooking.
        // mngf[u] = A x gf
        // mxv
//        bmv32_sparse_full_cc_512<int, int><<<grid_new, 512>>>(tA, grandparent, min_neighbor_parent_temp, bsrRowPtr, bsrColInd, nblockrows);
        bmv32_sparse_full_cc_new<int, int><<<grid_new, 1024>>>(tA, grandparent, min_neighbor_parent_temp, bsrRowPtr, bsrColInd, nblockrows);
//        printResVec<int><<<1,1>>>(min_neighbor_parent_temp, nrows);
        ewiseMin<<<(int)ceil(nrows/1024.0), 1024>>>(min_neighbor_parent, nrows, min_neighbor_parent, min_neighbor_parent_temp);
//        printResVec<int><<<1,1>>>(min_neighbor_parent, nrows);

        // f[f[u]] = mngf[u]
        // assignscatter /*TODO: change to gblast kernel */
        assignScatter<<<(int)ceil(nrows/1024.0), 1024>>>(parent, nrows, parent_temp, min_neighbor_parent);
//        printResVec<int><<<1,1>>>(parent, nrows);

        // 2) Aggressive hooking.
        // f = min(f, mngf)
        ewiseMin<<<(int)ceil(nrows/1024.0), 1024>>>(parent, nrows, parent, min_neighbor_parent);
//        printResVec<int><<<1,1>>>(parent, nrows);

        // 3) Shortcutting.
        // f = min(f, gf)
        ewiseMin<<<(int)ceil(nrows/1024.0), 1024>>>(parent, nrows, parent, parent_temp);
//        printResVec<int><<<1,1>>>(parent, nrows);

        // 4) Calculate grandparents.
        // gf[u] = f[f[u]]
        // extractgather /* TODO: change to gblast kernel */
        extractGather<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent, nrows, parent, parent);
//        printResVec<int><<<1,1>>>(grandparent, nrows);

        // 5) Check termination
        ewiseNotEqual<<<(int)ceil(nrows/1024.0), 1024>>>(diff, nrows, grandparent_temp, grandparent);
//        printResVec<int><<<1,1>>>(diff, nrows);

        // reduce
        resetSuccptr<<<1,1>>>(succptr);
        reduceAddInt<<<(int)ceil(nrows/1024.0), 1024>>>(succptr, nrows, diff);
        cudaMemcpy(&succ, succptr, sizeof(int), cudaMemcpyDeviceToHost);

        if (succ == 0) break;

        // grandparent_temp = grandparent
        initVec<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent_temp, nrows);
        cudaMemcpy(grandparent_temp, grandparent, nrows * sizeof(int), cudaMemcpyDeviceToDevice);

        // 6) Similar to BFS and SSSP, we should filter out the unproductive
        // vertices from the next iteration.
        // assign
//        printResVec<int><<<1,1>>>(grandparent, nrows);
        assignMax<<<(int)ceil(nrows/1024.0), 1024>>>(grandparent, nrows, diff, 2147483647);
//        printResVec<int><<<1,1>>>(grandparent, nrows);

        // copy result to output
        // v = parent, v is a vector

        printf("succ: %d\n", succ);

//        int k;
//        std::cin >> k;
    }

    bmvbin_timer.Stop();
    bmvbin32_time = bmvbin_timer.ElapsedMillis();

    printf("------------------------------------\n");
    int niter = iter; printf("niter: %d\n", niter);
    // ------

    // result verification /* TODO: change to gblast verify function */
    if (niter > 1) {
        int* h_parent = (int*) malloc(sizeof(int) * nrows);
        cudaMemcpy(h_parent, parent, nrows * sizeof(int), cudaMemcpyDeviceToHost);
        std::vector<int> h_cc_gpu(h_parent, h_parent + nrows);
        SimpleVerifyCc(nrows, h_csrRowPtr, h_csrColInd, h_cc_gpu, true/*suppres_zero*/);
    }

    // free storage
    cudaFree(tA);

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
#elif BLOCKSIZE == 16
    main16(argc, argv);
#elif BLOCKSIZE == 8
    main8(argc, argv);
#else
    main4(argc, argv);
#endif
}
