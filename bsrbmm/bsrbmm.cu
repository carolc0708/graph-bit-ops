#include <stdio.h>
#include <assert.h>

typedef unsigned long long ullong;

// A faster way to obtain lane id in a warp
#define GET_LANEID unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));

//For higher memory access efficiency
template <typename T>
__device__ __inline__ void store64(const void* addr, T a, T b)
{
    *((float2*)addr) = make_float2(*(float*)(&a),*(float*)(&b));
}

//For higher memory access efficiency
template <typename T>
__device__ __inline__ void store128(const void* addr, T a, T b, T c, T d)
{
    *((float4*)addr) = make_float4(*(float*)(&a),*(float*)(&b),*(float*)(&c),*(float*)(&d));
}

// C = A * A^T => col-major(A) * col-major(A) using rowbyrow model
// col-major packing bit 32
template <typename T>
__global__ void ToBit32Col(const T* __restrict__ A, unsigned* B, const int A_height, const int A_width) // blocksize, nblocks * blocksize
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // nblocks
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
#pragma unroll
    for (int i=0; i<32; i++)
    {
        T f0 = A[by*32*32+i*32+laneid];
        unsigned r0 = __brev(__ballot(f0>0));
        if (laneid == i) Bval = r0;
    }
    B[by*32+laneid] = Bval;
}

// row-major packing bit 32
template <typename T>
__global__ void ToBit32Row(const T* __restrict__ A, unsigned* B, const int A_height, const int A_width)
{
    const unsigned bx = blockIdx.x; // blockrows
    const unsigned by = blockIdx.y; // 1
    unsigned Bval=0;
#pragma unroll
    for (int i=0; i<32; i++)
    {
        T f0 = A[bx*32+i];
        Bval = (Bval<<1) + (f0>0);
    }
    B[bx] = Bval;
}

// col-major packing bit 64
template <typename T>
__global__ void ToBit64Col(const T* __restrict__ A, ullong* B, const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; //nblocks
    const unsigned bx = blockIdx.x; // 2 <- set this
    ullong Bval;
#pragma unroll
    for (int i=0; i<32; i++)
    {
        T f0 = A[by*64*64+bx*64*32+i*64+laneid]; //
        T f1 = A[by*64*64+bx*64*32+i*64+32+laneid]; //
        unsigned r0 = __ballot(f0>0);
        unsigned r1 = __ballot(f1>0);
        ullong l0;
        asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r0),"r"(r1)); //lo,hi
        if (laneid == i) Bval = __brevll(l0);
    }
    B[by*64+bx*32+laneid] = Bval;
}

// row-major packing bit 64
template <typename T>
__global__ void ToBit64Row(const T* __restrict__  A, ullong* B, const int A_height, const int A_width)
{
    GET_LANEID;
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    ullong Bval = 0;
#pragma unroll
    for (int i=0; i<64; i++)
    {
        T f0 = A[bx*64+i];
        Bval = (Bval<<1) | (f0>0);
    }
    B[bx] = Bval;
}

// bsr bmm32 no padding
// Cik = Sum(A_ij * B_jk)
// A (bsr matrix) * B (bsr matrix) = C (one float number)
// col-bin(32 x (blocksize x nblocks)) x col-bin(32 x (blocksize x nblocks))
template <typename Index, typename T>
__global__ void bmm32_sparse(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
            const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
            const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
            const Index nblockrows, const Index nblocks, const int nrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {

        GET_LANEID;
        T* Csub = &C[0];

        register int Cm[32] = {0};
        int sum = 0;

        // load
        int A_row_start = A_rowptr[bx]; // 0 32 64 . . . 991
        int A_row_end = A_rowptr[bx+1]; // 32 64 96 . . . 991 1022
        const unsigned* Asub = &(A[A_row_start*32]); // block is in continuous layout
        for (int i=A_row_start; i<A_row_end; i++) {
            unsigned r0 = Asub[(i-A_row_start)*32+laneid]; // <--

            int A_col = A_colind[i];
            int B_row_start = B_rowptr[A_col];
            int B_row_end = B_rowptr[A_col+1];
            const unsigned* Bsub = &(B[B_row_start*32]);
            for (int j=B_row_start; j<B_row_end; j++) {
                unsigned r1 = Bsub[(j-B_row_start)*32+laneid]; // <--
//                int B_col = B_colind[j];


                /* bmm */
                #pragma unroll
                for (int k=0; k<32; k++)
                {
                    unsigned r2 = __shfl(r1, k); //from lane-j, r1 of matrix B
//                    if (bx*32+laneid < nrows && B_col*32+k < nrows)
                        Cm[k] += __popc(r0 & r2); // each lane dot-product with the column of B
                }
                /* bmm */

            } // j in [B_row_start ... B_row_end]
        } // i in [A_row_start ... A_row_end]

        // store
//        __syncthreads();
        for (int l=0; l<32; l++) sum += Cm[l];
        atomicAdd(Csub+bx, sum);
//        __syncthreads();

    } // if bx < nblockrows + 1
}

// <-- not implemented
// bsr bmv64 no padding
// A (bsr matrix) * B (vector) = C (vector)
template <typename Index, typename T>
__global__ void bmm64_sparse(const ullong* __restrict__ A, const ullong* __restrict__ B, T* C,
                            const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
                            const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
                            const Index nblockrows, const Index nblocks, const int nrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {

        GET_LANEID;
        T* Csub = &C[0];

        register int Cm[64] = {0};
        int sum = 0;

        // load
        int A_row_start = A_rowptr[bx]; // 0 32 64 . . . 991
        int A_row_end = A_rowptr[bx+1]; // 32 64 96 . . . 991 1022
        const ullong* Asub = &(A[A_row_start*64]); // block is in continuous layout
        for (int i=A_row_start; i<A_row_end; i++) {
            ullong a0 = Asub[(i-A_row_start)*64+laneid];
            ullong a1 = Asub[(i-A_row_start)*64+32+laneid];

            int A_col = A_colind[i];
            int B_row_start = B_rowptr[A_col];
            int B_row_end = B_rowptr[A_col+1];
            const ullong* Bsub = &(B[B_row_start*64]);
            for (int j=B_row_start; j<B_row_end; j++) {
                ullong b0 = Bsub[(j-B_row_start)*64+laneid];
                ullong b1 = Bsub[(j-B_row_start)*64+32+laneid];
//                int B_col = B_colind[j];

                /* bmm */
                #pragma unroll
                for (int k=0; k<32; k++)
                {
                    ullong l0 = __shfl(b0,k);
                    ullong l1 = __shfl(b1,k);
//                    if (bx*64+laneid < nrows && B_col*64+k < nrows)
                        Cm[k] += __popcll(a0&l0) + __popcll(a1&l0);
//                    if (bx*64+laneid < nrows && B_col*64+32+k < nrows)
                        Cm[32+k] += __popcll(a0&l1) + __popcll(a1&l1);

                }
                /* bmm */

            } // j in [B_row_start ... B_row_end]
        } // i in [A_row_start ... A_row_end]

        // store
//        __syncthreads();
        for (int l=0; l<64; l++) sum += Cm[l];
        atomicAdd(Csub+bx, sum);
//        __syncthreads();

    } // if bx < nblockrows + 1
}

/**
* Extended Function: tril for csr
* Zeroes out matrix above main diagonal
* Note: for bsr, call it before csr2bsr
*/
// C = tril(A)
template <typename Index, typename T>
__global__ void tril_csr(const Index* A_rowptr, const Index* A_colind, const T* A_csrval,
                         const Index A_nrows, const Index A_nnz,
                         Index* C_rowptr, Index* C_colind, T* C_csrval, Index* C_nnz)
{
    Index remove = 0;
    for (Index row = 0; row < A_nrows; ++row) {
        Index edge_start = A_rowptr[row];
        Index edge_end = A_rowptr[row+1];

        // csrRowPtr_ update must only be done after row loads edge_start
        C_rowptr[row] -= remove;

        for (Index edge = edge_start; edge < edge_end; ++edge) {
            Index col = A_colind[edge];
            if (row < col) {
              remove++;
            } else {
              C_colind[edge-remove] = col;
              C_csrval[edge-remove] = A_csrval[edge];
            }
        }
    }
    // csrRowPtr_ update must be done for last element, which is equivalent to
    // nvals_
    C_rowptr[A_nrows] -= remove;
    C_nnz[0] = A_nnz - remove;
}

/* Extended method, non optimized */
template <typename T>
__global__ void reuduceSum(const T *gArr, const int arraySize, int *gOut) {
    int sum = 0;
    for (int i=0; i<arraySize; i++) {
        sum += (int)gArr[i];
    }
    *gOut = sum;
//    printf("sum: %d\n", *gOut);
}

/* Extended method to set diagonal elem to 0 */
__global__ void removeDiagonalNnz(const int* rowptr, const int* colind, float* csrval, const int nrows) {
    for(int i=0; i<nrows; i++) {
        for(int j=rowptr[i]; j<rowptr[i+1]; j++) {
            if (colind[j] == i) csrval[j] = 0.0;
        }
    }
}

/* sparsity mask */
// something like this
// printout to debug
__global__ void maskReduceSum(const int* mask_rowptr, const int* mask_colind, const int* rowptr, const int* colind, const float* csrval, const int nrows, float *gOut) {
    gOut[0] = (float)0;
    for(int i=0; i<nrows; i++) {
        for(int j=mask_rowptr[i]; j<mask_rowptr[i+1]; j++) {
            if (mask_colind[j] == colind[j]) gOut[0] += csrval[j];
        }
    }
}

/* For debug, gather result nnz by row */
__global__ void gatherNnzbyBlockrow(const int* rowptr, const int* colind, float* csrval,
                                  const int nrows, const int nblockrows, const int blocksize,
                                  float* resvec) {

    for (int i=0; i<nblockrows; i++) {
        int rowendpoint = (i+1)*blocksize < nrows ? (i+1)*blocksize : nrows;
        for(int j=rowptr[i*blocksize]; j<rowptr[rowendpoint]; j++) {
            resvec[i] += csrval[j];
        }
    }
}

