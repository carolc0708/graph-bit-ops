#include <stdio.h>
#include <assert.h>
#include <sys/time.h>

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

// to print unsigned
void bin(unsigned n)
{
    unsigned i;
    for (i = 1 << 31; i > 0; i = i / 2)
        (n & i) ? printf("1") : printf("0");
}

// weight should be col-major packing, layout is 32 * (32*numofblocks)
// input should be row-major packing, layout is whatever it is originally

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
// A (bsr matrix) * B (vector) = C (vector)
// col-bin(32 x (blocksize x nblocks)) * row-bin((nblockrows x nblocks) x 1) = (nblockrow x nblocks) x 1
template <typename Index, typename T>
__global__ void bmm32_sparse(const unsigned* __restrict__ A, const unsigned* __restrict__ B,
            T* C, const int A_height, const int A_width, const int B_width,
            const Index* __restrict__ rowptr, const Index* __restrict__ colind,
            const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y + blockIdx.z;
    if (bx < nblockrows + 1) {
        GET_LANEID;

        // load
        int row_start = rowptr[bx]; // 0 32 64 . . . 991
        int row_end = rowptr[bx+1]; // 32 64 96 . . . 991 1022

        const unsigned* Asub = &(A[row_start*32]); // block is in continuous layout
        const unsigned* Bsub = &(B[0]); // 0, when it is mv
        T* Csub = &(C[bx*32]);
        register unsigned Cm[1] = {0}; // allocate 1 register

        // compute
        // if that row has more than 1 col block
        for (int i=row_start; i<row_end; i++) {
            Cm[0] = 0;
            unsigned r0 = Asub[(i-row_start)*32+laneid]; // block is in continuous layout
            unsigned r1 = Bsub[(colind[i])]; // only first row is required

            Cm[0] += __popc(r0 & r1);
            // store
            Csub[laneid] += (T)(Cm[0]); //Csub[laneid] = (T)(Cm[0]>0);
        }
    }
}

// bsr bmv64 no padding
// A (bsr matrix) * B (vector) = C (vector)
template <typename Index, typename T>
__global__ void bmm64_sparse(const ullong* __restrict__ A, const ullong* __restrict__ B,
                            T* C, const int A_height, const int A_width, const int B_width,
                            const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                            const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y + blockIdx.z;
    if (bx < nblockrows + 1) {
        GET_LANEID;

        // load
        unsigned row_start = rowptr[bx];
        unsigned row_end = rowptr[bx+1];
        const ullong* Asub = &(A[row_start*64]);
        const ullong* Bsub = &(B[0]);
        T* Csub = &(C[bx*64]);
        register unsigned Cm[1] = {0};

        // compute
        for (int i=row_start; i<row_end; i++) {
            Cm[0] = 0;
            ullong a0 = Asub[(i-row_start)*64+laneid];
            ullong a1 = Asub[(i-row_start)*64+32+laneid];
            ullong b0 = Bsub[colind[i]];

            Cm[0] += (__popcll(a0 & b0) << 16) + __popcll(a1 & b0);

            // store
            short t0, t1;
            asm volatile("mov.b32 {%0,%1}, %2;":"=h"(t1),"=h"(t0):"r"(Cm[0]));
            Csub[laneid] += (T)t0;
            Csub[laneid+32] += (T)t1;
        }
    }
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

