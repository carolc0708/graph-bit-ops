#include <stdio.h>
#include <assert.h>

typedef unsigned char uchar; // 8
typedef unsigned short ushort; // 16
typedef unsigned long long ullong; // 64

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

// col-major packing bit 8
// process 4 8x8x4 at the same time
template <typename T>
__global__ void ToBit8Col(const T* __restrict__ A, uchar* B, const int nblocks)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // ceil(nblocks/16)
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
#pragma unroll
    for (int i=0; i<32; i++)
    {
        T f0 = by*8*8*4*4+i*32+laneid < nblocks*8*8 ? A[by*8*8*4*4+i*32+laneid] : 0; // <-- laneid will get consecutive 32 (half-block)
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>0?1:0)); //__brev(__ballot(f0>0));

        if (laneid == i) Bval = r0;
    }

    B[by*8*4*4+(laneid/2)*8+laneid%2*4] = (Bval & 0xFF000000) >> 24;
    B[by*8*4*4+(laneid/2)*8+laneid%2*4+1] = (Bval & 0x00FF0000) >> 16;
    B[by*8*4*4+(laneid/2)*8+laneid%2*4+2] = (Bval & 0x0000FF00) >> 8;
    B[by*8*4*4+(laneid/2)*8+laneid%2*4+3] = Bval & 0x000000FF;
}

// row-major packing bit 8
template <typename T>
__global__ void ToBit8Row(const T* __restrict__ A, uchar* B, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < (int)ceil((float)nblockrows/4)) {
        unsigned Bval=0;

        #pragma unroll
        for (int i=0; i<32; i++)
        {
            T f0 = A[bx*8*4+i];
            Bval = (Bval<<1) + (f0>0);
        }
        B[bx*4] = (Bval & 0xFF000000) >> 24;
        B[bx*4+1] = (Bval & 0x00FF0000) >> 16;
        B[bx*4+2] = (Bval & 0x0000FF00) >> 8;
        B[bx*4+3] = Bval & 0x000000FF;
    }
}

// col-major packing bit 16
template <typename T>
__global__ void ToBit16Col(const T* __restrict__ A, ushort* B, const int nblocks)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // ceil(nblocks/4)
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
#pragma unroll
    for (int i=0; i<32; i++)
    {
        T f0 = by*16*16*4+i*16*2+laneid < nblocks*16*16 ? A[by*16*16*4+i*16*2+laneid] : 0;
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>0?1:0)); //__brev(__ballot(f0>0));

        if (laneid == i) Bval = r0;
    }

    B[by*16*4+laneid*2] = (Bval & 0xFFFF0000) >> 16;
    B[by*16*4+laneid*2+1] = (Bval & 0x0000FFFF);
}
// 4 16x16 at the same time

// row-major packing bit 16
template <typename T>
__global__ void ToBit16Row(const T* __restrict__ A, ushort* B, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < (int)ceil((float)nblockrows/2)) {
        unsigned Bval=0;
#pragma unroll
        for (int i=0; i<32; i++)
        {
            T f0 = A[bx*32+i];
            Bval = (Bval<<1) + (f0>0);
        }

        B[bx*2] = (Bval & 0xFFFF0000) >> 16;
        B[bx*2+1] = (Bval & 0x0000FFFF);
    }
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
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>0?1:0));//__brev(__ballot(f0>0));
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
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>0?1:0)); //__ballot(f0>0);
        unsigned r1 = __brev(__ballot_sync(0xFFFFFFFF, f1>0?1:0)); //__ballot(f1>0);
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

template <typename Index, typename T>
__global__ void bmm8_sparse(const uchar* __restrict__ A, const uchar* __restrict__ B, T* C,
                            const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
                            const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
                            const Index nblockrows, const Index nblocks, const int nrows, int* runtime)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {
#ifdef PROF
        clock_t start_time = clock();
#endif
        GET_LANEID;
        T* Csub = &C[0];

        register int Cm[8] = {0};
        int sum = 0;

        // load
        int A_row_start = A_rowptr[bx];
        int A_row_end = A_rowptr[bx+1];
        const uchar* Asub = &(A[A_row_start*8]);
        for (int i=A_row_start; i<(int)ceil(float(A_row_end)/4)*4; i+=4) {
            uchar r0 = 0;
            if (i*8+laneid < A_row_end*8) {
                r0 = Asub[(i-A_row_start)*8+laneid];

                int A_col = A_colind[i+laneid/8];
                int B_row_start = B_rowptr[A_col];
                int B_row_end = B_rowptr[A_col+1];
                const uchar* Bsub = &(B[B_row_start*8]);
                for (int j=B_row_start; j<B_row_end; j+=1) {
                    uchar r1 = Bsub[(j-B_row_start)*8+laneid%8];
    //                int B_col = B_colind[j];

                    /* bmm */
                    #pragma unroll
                    for (int k=0; k<8; k++)
                    {
                        uchar r2 = __shfl_sync(0xFFFFFFFF, r1, k+(laneid/8)*8); //__shfl(r1, k+(laneid/8)*8);
                        Cm[k] += __popc(r0 & r2);
                    }
                    /* bmm */
                } // j in [B_row_start ... B_row_end]
            } // if i*8+laneid < A_row_end*8
        } // i in [A_row_start ... A_row_end]

        // store

        for (int l=0; l<8; l++) sum += Cm[l];

        atomicAdd(Csub, sum);

//        __shared__ int result[32];
//        result[laneid] = sum;
//
//        __syncthreads();
//
//        if (laneid == 0) {
//            for(int i=0; i<32; i++) {
//                Csub[bx] += result[i];
//            }
//        }


#ifdef PROF
        clock_t stop_time = clock();
        runtime[bx] = (int)(stop_time - start_time);
//        printf("[%d, %d] %d ", bx, laneid, (int)(stop_time - start_time));
#endif

    } // if bx < nblockrows
}

template <typename Index, typename T>
__global__ void bmm16_sparse(const ushort* __restrict__ A, const ushort* __restrict__ B, T* C,
            const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
            const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
            const Index nblockrows, const Index nblocks, const int nrows, int* runtime)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {
#ifdef PROF
        clock_t start_time = clock();
#endif
        GET_LANEID;
        T* Csub = &C[0];

        register int Cm[16] = {0};
        int sum = 0;

        // load
        int A_row_start = A_rowptr[bx];
        int A_row_end = A_rowptr[bx+1];
        const ushort* Asub = &(A[A_row_start*16]);
        for (int i=A_row_start; i<(int)ceil(float(A_row_end)/2)*2; i+=2) {
            ushort r0 = 0;

            if (i*16+laneid < A_row_end*16) {
                r0 = Asub[(i-A_row_start)*16+laneid];

                int A_col = A_colind[i+laneid/16];
                int B_row_start = B_rowptr[A_col];
                int B_row_end = B_rowptr[A_col+1];
                const ushort* Bsub = &(B[B_row_start*16]);
                for (int j=B_row_start; j<B_row_end; j+=1) {
                    ushort r1 = Bsub[(j-B_row_start)*16+laneid%16];
    //                int B_col = B_colind[j];

                    /* bmm */
                    #pragma unroll
                    for (int k=0; k<16; k++)
                    {
                        ushort r2 = __shfl_sync(0xFFFFFFFF, r1, k+(laneid/16)*16); //__shfl(r1, k+(laneid/16)*16);
                        Cm[k] += __popc(r0 & r2);
                    }
                    /* bmm */

                } // j in [B_row_start ... B_row_end]
            } // i*16+laneid < A_row_end*16
        } // i in [A_row_start ... A_row_end]

        // store
        for (int l=0; l<16; l++) sum += Cm[l];
        atomicAdd(Csub, sum);

#ifdef PROF
        clock_t stop_time = clock();
        runtime[bx] = (int)(stop_time - start_time);
//        printf("[%d, %d] %d ", bx, laneid, (int)(stop_time - start_time));
#endif

    } // if bx < nblockrows
}

// bsr bmm32 no padding
// Cik = Sum(A_ij * B_jk)
// A (bsr matrix) * B (bsr matrix) = C (one float number)
// col-bin(32 x (blocksize x nblocks)) x col-bin(32 x (blocksize x nblocks))
template <typename Index, typename T>
__global__ void bmm32_sparse(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
            const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
            const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
            const Index nblockrows, const Index nblocks, const int nrows, int* runtime)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {
#ifdef PROF
        clock_t start_time = clock();
#endif
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
                    unsigned r2 = __shfl_sync(0xFFFFFFFF, r1, k); //__shfl(r1, k); //from lane-j, r1 of matrix B
                    Cm[k] += __popc(r0 & r2); // each lane dot-product with the column of B
                }
                /* bmm */

            } // j in [B_row_start ... B_row_end]
        } // i in [A_row_start ... A_row_end]

        // store
        for (int l=0; l<32; l++) sum += Cm[l];
        atomicAdd(Csub, sum);

#ifdef PROF
        clock_t stop_time = clock();
        runtime[bx] = (int)(stop_time - start_time);
//        printf("[%d, %d] %d ", bx, laneid, (int)(stop_time - start_time));
#endif

    } // if bx < nblockrows + 1
}

// <-- not implemented
// bsr bmv64 no padding
// A (bsr matrix) * B (vector) = C (vector)
template <typename Index, typename T>
__global__ void bmm64_sparse(const ullong* __restrict__ A, const ullong* __restrict__ B, T* C,
                            const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
                            const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
                            const Index nblockrows, const Index nblocks, const int nrows, int* runtime)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {
#ifdef PROF
        clock_t start_time = clock();
#endif

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
                    ullong l0 = __shfl_sync(0xFFFFFFFF, b0, k); //__shfl(b0,k);
                    ullong l1 = __shfl_sync(0xFFFFFFFF, b1, k); //__shfl(b1,k);

                    Cm[k] += __popcll(a0&l0) + __popcll(a1&l0);
                    Cm[32+k] += __popcll(a0&l1) + __popcll(a1&l1);

                }
                /* bmm */

            } // j in [B_row_start ... B_row_end]
        } // i in [A_row_start ... A_row_end]

        // store
        for (int l=0; l<64; l++) sum += Cm[l];
        atomicAdd(Csub, sum);

#ifdef PROF
        clock_t stop_time = clock();
        runtime[bx] = (int)(stop_time - start_time);
//        printf("[%d, %d] %d ", bx, laneid, (int)(stop_time - start_time));
#endif
    } // if bx < nblockrows + 1
}

//======================================================================================
// Preprocessing and Postprocessing function
//======================================================================================

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

//======================================================================================
// Function for workload distribution
//======================================================================================
// naive 32
template <typename Index, typename T>
__global__ void bmm32_sparse_workloadmergeNsplit(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
                                                const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
                                                const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
                                                const Index* __restrict__ workload_info_list, const Index* __restrict__ workload_size_list_acc,
                                                const Index workloadsize, int* runtime, int* load)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < workloadsize) {
#ifdef PROF
        clock_t start_time = clock();
#endif

    // set metadata
    int list_start = workload_size_list_acc[bx], list_end = workload_size_list_acc[bx+1];
    int row = workload_info_list[list_start];
    int row_start = workload_info_list[list_start+1];
    int numworkload;
    if (list_end - list_start == 3) { // only 1 blockrow
        numworkload = 1;
    } else { // more than 1 blockrows
        numworkload = list_end - list_start - 2;
    }

    // load
    GET_LANEID;
    T* Csub = &C[0];

    int workload = 0;
    for (int w=0; w<numworkload; w++) {
        // set pointer
        row_start += workload; // move 1 step
        workload = workload_info_list[list_start+2+w]; // get workload

        // load location
        const unsigned* Asub = &(A[row_start*32]);

        // compute
        register int Cm[32] = {0};
        int sum = 0;
        for (int i=row_start; i<row_start+workload; i++) {
            unsigned r0 = Asub[(i-row_start)*32+laneid]; // <--

            int A_col = A_colind[i];
            int B_row_start = B_rowptr[A_col];
            int B_row_end = B_rowptr[A_col+1];
            const unsigned* Bsub = &(B[B_row_start*32]);
            for (int j=B_row_start; j<B_row_end; j++) {
                unsigned r1 = Bsub[(j-B_row_start)*32+laneid];

                /* bmm */
                #pragma unroll
                for (int k=0; k<32; k++)
                {
                    unsigned r2 = __shfl_sync(0xFFFFFFFF, r1, k); //__shfl(r1, k); //from lane-j, r1 of matrix B
                    Cm[k] += __popc(r0 & r2); // each lane dot-product with the column of B
                }
                /* bmm */

            } // j in [B_row_start ... B_row_end]
        } // i in [A_row_start ... A_row_end]

        // store
        for (int l=0; l<32; l++) sum += Cm[l];
        atomicAdd(Csub, sum);

    } // w < numworkload



#ifdef PROF
        clock_t stop_time = clock();
        runtime[bx] = (int)(stop_time - start_time);
        load[bx] = row_start + workload - workload_info_list[list_start+1]; // temp
//        GET_LANEID;
//        if (laneid == 1 && load[bx] == 0) {printf("[%d] %d %d\n", bx, (int)(stop_time - start_time), (int)(row_end-row_start));}
#endif
    }
}

// split_merge
__global__ void count_workload_merge_and_split(int* workloadsize, int* rowptr, const int nblockrows, const int* colind, const int MAX) {

    int cnt = 0;
    int rest = 0;

    for(int i=0; i<nblockrows; i++) {
        int load = rowptr[i+1]-rowptr[i];
        if (load == 0) continue;

        if (rest) { load = rest; i -= 1; rest = 0;} // set back to previous row

        if (load < MAX) { // merge
            int temp = 0;
            while (temp + load <= MAX && i < nblockrows) { // when it is last
                temp += load;
                i += 1;

                load = i < nblockrows ? rowptr[i+1]-rowptr[i] : 0; // force leave
            }
            i -= 1;
            cnt += 1;

        } else if (load > MAX) { // split
            int n = (int)floorf((float)load / MAX);
            cnt += n;

            // the last one will be merge with the following
            rest = load % MAX;

        } else { // preserve
            cnt += 1;
        }
    }

    if (rest) cnt += 1;

    *workloadsize = cnt;

}

// function to get metadata for computation
// get workload info for bmm (different from bmv)
// should consider the load (seen as weight) of each col
// split point can be at:
// (1) blockrows' load (like bmv)
// (2) blockrows' col (weight/non-weight)
// (3) blockrows' col's load
__global__ void getWorkloadInfo (const int* __restrict__ rowptr, const int nblockrows, const int MAX,
                                 int* workload_size_list, int* workload_info_list, const int workloadsize, int* workload_info_list_size)
{
    int cnt = 0; int wid = 0;
    int rest = 0;

    int row_start, row_end, load;
    int i = 0;
    for(i=0; i<nblockrows; i++) {
        row_start = rowptr[i]; row_end = rowptr[i+1];
        load = row_end - row_start;

        if (load == 0) continue;

        if (rest) { load = rest; i -= 1; rest = 0; row_start = rowptr[i] + (rowptr[i+1]-rowptr[i])/MAX * MAX;} // set back to previous row

        if (load < MAX) { // merge

            workload_info_list[cnt++] = i; // row
            workload_info_list[cnt++] = row_start; // row_start

            int temp = 0; int mcnt = 0;
            while (temp + load <= MAX && i < nblockrows) {
                temp += load;
                workload_info_list[cnt++] = load; // workload
                mcnt += 1;
                i += 1;

                // next iter
                load = i < nblockrows ? rowptr[i+1]-rowptr[i] : 0; // force leave
            }
            i -= 1;

            workload_size_list[wid++] = mcnt;

        } else if (load > MAX) { // split

            int n = load / MAX;
            for(int j=0; j<n; j++) {
                workload_size_list[wid++] = 1;
                workload_info_list[cnt++] = i; // row
                workload_info_list[cnt++] = row_start + j * MAX; // row_start
                workload_info_list[cnt++] = MAX; // workload
            }

            // the last one will be merge with the following
            rest = load % MAX;

        } else { // preserve (load == MAX)

            workload_size_list[wid++] = 1;
            workload_info_list[cnt++] = i; // row
            workload_info_list[cnt++] = row_start; // row_start
            workload_info_list[cnt++] = MAX; // workload

        }
    }

    if (rest) {
        workload_size_list[wid++] = 1;
        workload_info_list[cnt++] = i-1; // row
        workload_info_list[cnt++] = row_start + load/MAX * MAX; // row_start
        workload_info_list[cnt++] = rest; // workload
    }

    *workload_info_list_size = cnt;

    // workload_size_list
    // an array of sublist_size+2, having [row, row_start,  workload0, workload1 ...]

}
