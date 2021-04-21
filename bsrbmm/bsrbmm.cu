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

// col-major packing bit 4
template <typename T>
__global__ void ToBit4Col(const T* __restrict__ A, uchar* B, const int nblocks)
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // ceil(nblocks/64)
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
    T f0;

#pragma unroll
    for (int i=0; i<32; i++)
    {
        f0 = by*16*64+i*16*2+laneid < nblocks*16 ? A[by*16*64+i*16*2+laneid] : 0; // <-- laneid will get consecutive 32 (2-blocks)
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>0?1:0)); //__brev(__ballot(f0>0));
        if (laneid == i) Bval = r0;
    }

    // layout block0 at high-16
    B[by*4*64+laneid*4*2] = (Bval & 0xF0000000) >> 28;
    B[by*4*64+laneid*4*2+1] = (Bval & 0x0F000000) >> 24;
    B[by*4*64+laneid*4*2+2] = (Bval & 0x00F00000) >> 20;
    B[by*4*64+laneid*4*2+3] = (Bval & 0x000F0000) >> 16;

    // layout block1 at low-16
    B[by*4*64+laneid*4*2+4] = (Bval & 0x0000F000) >> 12;
    B[by*4*64+laneid*4*2+5] = (Bval & 0x00000F00) >> 8;
    B[by*4*64+laneid*4*2+6] = (Bval & 0x000000F0) >> 4;
    B[by*4*64+laneid*4*2+7] = (Bval & 0x0000000F);
}

// row-major packing bit 4
template <typename T>
__global__ void ToBit4Row(const T* __restrict__ A, uchar* B, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < (int)ceil((float)nblockrows/4)) {
        unsigned Bval=0;
        T f0;

        #pragma unroll
        for (int i=0; i<32; i++)
        {
            if (i%8 < 4) f0 = (T)(0); // high-4 bit remain 0
            else f0 = A[bx*4*4+(i-4*((i/8)+1))];

            Bval = (Bval<<1) + (f0>0);
        }
        B[bx*4] = (Bval & 0xFF000000) >> 24;
        B[bx*4+1] = (Bval & 0x00FF0000) >> 16;
        B[bx*4+2] = (Bval & 0x0000FF00) >> 8;
        B[bx*4+3] = Bval & 0x000000FF;
    }
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
__global__ void bmm4_sparse(const uchar* __restrict__ A, const uchar* __restrict__ B, T* C,
                            const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
                            const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
                            const Index nblockrows, const Index nblocks, const int nrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {
        GET_LANEID;
        T* Csub = &C[0];

        register int Cm[4] = {0};
        int sum = 0;

        // load
        int A_row_start = A_rowptr[bx];
        int A_row_end = A_rowptr[bx+1];
        const uchar* Asub = &(A[A_row_start*4]);
        for (int i=A_row_start; i<(int)ceil(float(A_row_end)/8)*8; i+=8) {
            uchar r0 = 0;
            if (i*4+laneid < A_row_end*4) {
                r0 = Asub[(i-A_row_start)*4+laneid];

                int A_col = A_colind[i+laneid/4];
                int B_row_start = B_rowptr[A_col];
                int B_row_end = B_rowptr[A_col+1];
                const uchar* Bsub = &(B[B_row_start*4]);
                for (int j=B_row_start; j<B_row_end; j+=1) {
                    uchar r1 = Bsub[(j-B_row_start)*4+laneid%4];
    //                int B_col = B_colind[j];

                    /* bmm */
                    #pragma unroll
                    for (int k=0; k<4; k++)
                    {
                        uchar r2 = __shfl_sync(0xFFFFFFFF, r1, k+(laneid/4)*4); //__shfl(r1, k+(laneid/4)*4);
                        Cm[k] += __popc(r0 & r2);
                    }
                    /* bmm */
                } // j in [B_row_start ... B_row_end]
            } // if i*4+laneid < A_row_end*4
        } // i in [A_row_start ... A_row_end]

        // store

        for (int l=0; l<4; l++) sum += Cm[l];

        atomicAdd(Csub, sum);

    } // if bx < nblockrows
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
//                 if (laneid == 0) printf("C(%d, %d) += A(%d, %d) * B(%d, %d)\n", bx, B_colind[j], bx, A_col, A_col, B_colind[j]);

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
// Mask function
//======================================================================================
// Cik = Sum(A_ij * B_jk) * A_ik
template <typename Index, typename T>
__global__ void bmm32_sparse_masked(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
            const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
            const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
            const Index nblockrows, const Index nblocks, const int nrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {
        GET_LANEID;
        T* Csub = &C[0];

        int sum = 0;
        bool mask = false;
        unsigned m = 0;

        // load
        int A_row_start = A_rowptr[bx]; // 0 32 64 . . . 991
        int A_row_end = A_rowptr[bx+1]; // 32 64 96 . . . 991 1022
        const unsigned* Asub = &(A[A_row_start*32]); // block is in continuous layout
        for (int i=A_row_start; i<A_row_end; i++) {
            unsigned r0 = Asub[(i-A_row_start)*32+laneid];

            int A_col = A_colind[i];
            int B_row_start = B_rowptr[A_col];
            int B_row_end = B_rowptr[A_col+1];
            const unsigned* Bsub = &(B[B_row_start*32]);
            for (int j=B_row_start; j<B_row_end; j++) {

                /* checking mask */
                int B_col = B_colind[j];
                #pragma unroll
                for(int l=A_row_start; l<A_row_end; l++) { // A_ik
                    if (A_colind[l] == B_col) {
                        mask = true;
                        m = Asub[(l-A_row_start)*32+laneid];
                        break;
                    }
                }
                /* checking mask */

                if (mask) {
                    unsigned r1 = Bsub[(j-B_row_start)*32+laneid];
                    unsigned register Cm[32] = {0};

                    /* bmm */
                    #pragma unroll
                    for (int k=0; k<32; k++)
                    {
                        unsigned r2 = __shfl_sync(0xFFFFFFFF, r1, k);
                        Cm[k] = __popc(r0 & r2);
                        sum += (int)(((m>>(31-k))&0x1)?Cm[k]:0); // masking
                    }
                    /* bmm */

                    /* reset mask */
                    mask = false;
                }
            } // j in [B_row_start ... B_row_end]
        } // i in [A_row_start ... A_row_end]

        // store
        atomicAdd(Csub, sum);
    } // if bx < nblockrows + 1
}

// binary search
template <typename Index>
__device__ Index binarySearch(const Index* array,
                              Index        target,
                              Index        begin,
                              Index        end)
{
    while (begin < end) {
        int mid  = begin + (end - begin) / 2;
        int item = array[mid];
        if (item == target)
          return mid;
        bool larger = (item > target);
        if (larger)
          end = mid;
        else
          begin = mid + 1;
    }
    return -1;
}

// Cik = Sum(A_ij * B_jk) * A_ik
template <typename Index, typename T>
__global__ void bmm32_sparse_masked_v2(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
                                       const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
                                       const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
                                       const Index nblockrows, const Index nblocks, const int nrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {

        GET_LANEID;
        T* Csub = &C[0];
        int sum = 0;

        // load
        int A_row_start = A_rowptr[bx];
        int A_row_end = A_rowptr[bx+1];

        for (int edge=A_row_start; edge<A_row_end; edge++) { // iterate the mask block location
            unsigned mask_val = A[edge*32+laneid];

            if (mask_val) {
                register int Cm[32] = {0};

                // load B bounds on which we must do binary search
                Index B_ind = A_colind[edge]; // (0, 0), (0, 2)
                Index B_col_start = B_rowptr[B_ind];
                Index B_col_end = B_rowptr[B_ind+1];


                // do bmm along the candidate blocks
                for(int i=A_row_start; i<A_row_end; i++) {

                    Index A_col = A_colind[i];
                    Index B_row = binarySearch(B_colind, A_col, B_col_start, B_col_end);

                    unsigned r0 = A[i*32+laneid];

                    if (B_row != -1) {
                        unsigned r1 = B[B_row*32+laneid];

                        /* bmm */
                        #pragma unroll
                        for (int k=0; k<32; k++)
                        {
                            unsigned r2 = __shfl_sync(0xFFFFFFFF, r1, k);
                            Cm[k] += __popc(r0 & r2);
                        }
                        /* bmm */

                    } //B_row != -1
                } //i

                #pragma unroll
                for (int k=0; k<32; k++)
                {
                    sum += (int)(((mask_val>>(31-k))&0x1)?Cm[k]:0); // masking
                }
            } //mask_val != 0
        } //edge

        // store
        atomicAdd(Csub, sum);
    }
}

// Cik = Sum(A_ij * B_jk) * A_ik
template <typename Index, typename T>
__global__ void bmm32_sparse_masked_v3(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
                                       const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
                                       const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
                                       const Index nblockrows, const Index nblocks, const int nrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {

        GET_LANEID;
        T* Csub = &C[0];
        int sum = 0;
        register int Cm[32] = {0};


        // load
        int A_row_start = A_rowptr[bx];
        int A_row_end = A_rowptr[bx+1];

        for (int edge=A_row_start; edge<A_row_end; edge++) { // iterate the mask block location
            unsigned mask_val = A[edge*32+laneid];
            Index C_col = A_colind[edge]; // C(0, 0), C(0, 1)

            if (mask_val) {

                for(int A_col=A_row_start; A_col<A_row_end; A_col++) {
                    unsigned r0 = A[A_col*32+laneid];

                    Index B_ind = A_colind[A_col];
                    Index B_row_start = B_rowptr[B_ind];
                    Index B_row_end = B_rowptr[B_ind+1];
                    Index B_col = binarySearch(B_colind, C_col, B_row_start, B_row_end);

                    if (B_col != -1) {

    //                    if (laneid == 0) printf("C(%d, %d) += A(%d, %d) * B(%d, %d)\n", bx, C_col, bx, B_ind, B_ind, C_col);
                        unsigned r1 = B[B_col*32+laneid];

                        /* bmm */
                        #pragma unroll
                        for (int k=0; k<32; k++)
                        {
                            unsigned r2 = __shfl_sync(0xFFFFFFFF, r1, k);
                            Cm[k] = __popc(r0 & r2);
                            sum += (int)(((mask_val>>(31-k))&0x1)?Cm[k]:0);
                        }
                        /* bmm */
                    } //B_col != -1
                }
            } // masking
        } //edge

        // store
        atomicAdd(Csub, sum);
    }
}

// Cik = Sum(A_ij * B_jk) * A_ik
template <typename Index, typename T>
__global__ void bmm4_sparse_full(const uchar* __restrict__ A, const uchar* __restrict__ B, T* C,
                                   const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
                                   const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
                                   const Index* __restrict__ C_rowptr, const Index* __restrict__ C_colind,
                                   const Index nblockrows, const Index nblocks, const int nrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {
        GET_LANEID;
        T* Csub = &C[0];

        // load
        int A_row_start = A_rowptr[bx];
        int A_row_end = A_rowptr[bx+1];
        const uchar* Asub = &(A[A_row_start*4]);

        Index C_row_start = C_rowptr[bx*4+laneid%4];
        Index C_row_end = C_rowptr[bx*4+laneid%4+1];


        for (int i=A_row_start; i<(int)ceil(float(A_row_end)/8)*8; i+=8) {
            uchar r0 = 0;

            if (i*4+laneid < A_row_end*4) {


                register unsigned Cm[4] = {0};
                Index C_row = C_row_start;

                r0 = Asub[(i-A_row_start)*4+laneid];

                int A_col = A_colind[i+laneid/4];
                int B_row_start = B_rowptr[A_col];
                int B_row_end = B_rowptr[A_col+1];
                const uchar* Bsub = &(B[B_row_start*4]);
                for (int j=B_row_start; j<B_row_end; j+=1) {
                    uchar r1 = Bsub[(j-B_row_start)*4+laneid%4];
    //                int B_col = B_colind[j];

                    /* bmm */
                    #pragma unroll
                    for (int k=0; k<4; k++)
                    {
                        uchar r2 = __shfl_sync(0xFFFFFFFF, r1, k+(laneid/4)*4); //__shfl(r1, k+(laneid/4)*4);
                        Cm[k] = __popc(r0 & r2);
                    }
                    /* bmm */

                    // add to C_csrval
                    for (int l=0; l<4; l++) {
                        if (Cm[l])
                        {
                            atomicAdd(Csub+C_row, (T)Cm[l]); // 35.431 coAu
                            // Csub[C_row] += (T)Cm[l]; // 81.818
                            C_row++;
                        }
                    }

                } // j in [B_row_start ... B_row_end]
            } // i*4+laneid < A_row_end*4
        } // i in [A_row_start ... A_row_end]
    } // bx < nblockrow
}


// Cik = Sum(A_ij * B_jk) * A_ik
template <typename Index, typename T>
__global__ void bmm32_sparse_full_v1(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
                                   const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
                                   const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
                                   const Index* __restrict__ C_rowptr, const Index* __restrict__ C_colind,
                                   const Index nblockrows, const Index nblocks, const int nrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {
        GET_LANEID;
        T* Csub = &C[0];

        // load
        int A_row_start = A_rowptr[bx]; // 0 32 64 . . . 991
        int A_row_end = A_rowptr[bx+1]; // 32 64 96 . . . 991 1022
        const unsigned* Asub = &(A[A_row_start*32]); // block is in continuous layout

        Index C_row_start = C_rowptr[bx*32+laneid];
        Index C_row_end = C_rowptr[bx*32+laneid+1];

        for (int i=A_row_start; i<A_row_end; i++) {
            unsigned r0 = Asub[(i-A_row_start)*32+laneid]; // <--
            Index C_row = C_row_start;

            int A_col = A_colind[i];
            int B_row_start = B_rowptr[A_col];
            int B_row_end = B_rowptr[A_col+1];
            const unsigned* Bsub = &(B[B_row_start*32]);
            for (int j=B_row_start; j<B_row_end; j++) {
                unsigned r1 = Bsub[(j-B_row_start)*32+laneid];
                int B_col = B_colind[j];

                register unsigned Cm[32] = {0};

                /* bmm */
                #pragma unroll
                for (int k=0; k<32; k++)
                {
                    unsigned r2 = __shfl_sync(0xFFFFFFFF, r1, k); //__shfl(r1, k); //from lane-j, r1 of matrix B
                    Cm[k] = __popc(r0 & r2); // each lane dot-product with the column of B
//                    if (Cm[k]) { atomicAdd(Csub+C_row, (T)Cm[k]); C_row++;} // 50.671, 83.306
                }
                /* bmm */

                // add to C_csrval
                for (int l=0; l<32; l++) {
                    if (Cm[l])
                        {
                            atomicAdd(Csub+C_row, (T)Cm[l]); // 35.431 coAu
    //                        Csub[C_row] += (T)Cm[l]; // 81.818
                            C_row++;
                        }
                }

            } // j in [B_row_start ... B_row_end]
        } // i in [A_row_start ... A_row_end]
    } // bx < nblockrow
}


//======================================================================================
// TC masked function
//======================================================================================
template <typename Index, typename T>
__global__ void bmm4_sparse_masked_v4(const uchar* __restrict__ A, const uchar* __restrict__ B, T* C,
                                   const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
                                   const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
                                   const Index nblockrows, const Index nblocks, const int nrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {
        GET_LANEID;
        T* Csub = &C[0];
        register T sum[1] = {0};

        // load
        int A_row_start = A_rowptr[bx];
        int A_row_end = A_rowptr[bx+1];
        const uchar* Asub = &(A[A_row_start*4]);


        for (int i=A_row_start; i<(int)ceil(float(A_row_end)/8)*8; i+=8) {
            uchar r0 = 0;

            if (i*4+laneid < A_row_end*4) {
               register unsigned Cm[4] = {0};

                r0 = Asub[(i-A_row_start)*4+laneid];

                int A_col = A_colind[i+laneid/4];
                int B_row_start = B_rowptr[A_col];
                int B_row_end = B_rowptr[A_col+1];
                const uchar* Bsub = &(B[B_row_start*4]);
                for (int j=B_row_start; j<B_row_end; j+=1) {
                    uchar r1 = Bsub[(j-B_row_start)*4+laneid%4];
                    int B_col = B_colind[j];

                    /* bmm */
                    #pragma unroll
                    for (int k=0; k<4; k++)
                    {
                        uchar r2 = __shfl_sync(0xFFFFFFFF, r1, k+(laneid/4)*4);
                        Cm[k] = __popc(r0 & r2);
                    }
                    /* bmm */


                    // load mask
                    Index mask_ind = binarySearch(A_colind, B_col, A_row_start, A_row_end);
                    if (mask_ind != -1) {
                        unsigned mask_val = Asub[(mask_ind-A_row_start)*4+laneid%4];

                        if (mask_val) {
//                            T sum = 0;
                            // add to C_csrval
                            for (int l=0; l<4; l++) {
                                sum[0] += (T)(((mask_val>>(3-l))&0x1)?Cm[l]:0);
                            }
//                            atomicAdd(Csub, sum);
                        }
                    } // mask_ind != -1

                } // j in [B_row_start ... B_row_end]
            } // i*4+laneid < A_row_end*4
        } // i in [A_row_start ... A_row_end]
        atomicAdd(Csub, sum[0]);
    } // bx < nblockrow
}

template <typename Index, typename T>
__global__ void bmm4_sparse_masked_v5(const uchar* __restrict__ A, const uchar* __restrict__ B, T* C,
                                   const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
                                   const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
                                   const Index nblockrows, const Index nblocks, const int nrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {
        GET_LANEID;
        T* Csub = &C[0];
        T sum = 0;

        int A_row_start = A_rowptr[bx];
        int A_row_end = A_rowptr[bx+1];

        for (Index i=A_row_start; i<(int)ceil(float(A_row_end)/8)*8; i+=8) {

            if (i*4+laneid < A_row_end*4) {
                Index mask_col = A_colind[i+laneid/4];
                uchar mask_val = A[i*4+laneid];  // process 8 blocks at a time
                register unsigned Cm[4] = {0};

                // collect the result of bmms
                for (Index j=A_row_start; j<A_row_end; j++) {
                    Index A_col = A_colind[j];
                    Index B_row_start = B_rowptr[A_col];
                    Index B_row_end = B_rowptr[A_col+1];

                    Index B_col = binarySearch(B_colind, mask_col, B_row_start, B_row_end);

                    if (B_col != -1) {
                        uchar r0 = A[j*4+laneid%4];
                        uchar r1 = B[B_col*4+laneid%4];

                        /* bmm */
                        #pragma unroll
                        for (int k=0; k<4; k++)
                        {
                            uchar r2 = __shfl_sync(0xFFFFFFFF, r1, k+(laneid/4)*4);
                            Cm[k] += (T)(__popc(r0 & r2));
                        }
                        /* bmm */
                    }
                }

                // do masking
                for (int l=0; l<4; l++)
                {
                    sum += (T)(((mask_val>>(3-l))&0x1)?Cm[l]:0);
                }

            }
        }

        atomicAdd(Csub, sum);
    }
}

template <typename Index, typename T>
__global__ void bmm8_sparse_masked_v4(const uchar* __restrict__ A, const uchar* __restrict__ B, T* C,
                                   const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
                                   const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
                                   const Index nblockrows, const Index nblocks, const int nrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {
        GET_LANEID;
        T* Csub = &C[0];
        register T sum[1] = {0};

        // load
        int A_row_start = A_rowptr[bx];
        int A_row_end = A_rowptr[bx+1];
        const uchar* Asub = &(A[A_row_start*8]);
        for (int i=A_row_start; i<(int)ceil(float(A_row_end)/4)*4; i+=4) {
            uchar r0 = 0;

            if (i*8+laneid < A_row_end*8) {
               register unsigned Cm[8] = {0};
                r0 = Asub[(i-A_row_start)*8+laneid];

                int A_col = A_colind[i+laneid/8];
                int B_row_start = B_rowptr[A_col];
                int B_row_end = B_rowptr[A_col+1];
                const uchar* Bsub = &(B[B_row_start*8]);
                for (int j=B_row_start; j<B_row_end; j+=1) {
                    uchar r1 = Bsub[(j-B_row_start)*8+laneid%8];
                    int B_col = B_colind[j];

                    /* bmm */
                    #pragma unroll
                    for (int k=0; k<8; k++)
                    {
                        uchar r2 = __shfl_sync(0xFFFFFFFF, r1, k+(laneid/8)*8); //__shfl(r1, k+(laneid/8)*8);
                        Cm[k] = __popc(r0 & r2);
                    }
                    /* bmm */


                    // load mask
                    Index mask_ind = binarySearch(A_colind, B_col, A_row_start, A_row_end);
                    if (mask_ind != -1) {
                        unsigned mask_val = Asub[(mask_ind-A_row_start)*8+laneid%8];

                        if (mask_val) {
//                            T sum = 0;
                            // add to C_csrval
                            for (int l=0; l<8; l++) {
                                sum[0] += (T)(((mask_val>>(7-l))&0x1)?Cm[l]:0);
                            }
//                            atomicAdd(Csub, sum);
                        }
                    } // mask_ind != -1

                } // j in [B_row_start ... B_row_end]
            } // i*4+laneid < A_row_end*4
        } // i in [A_row_start ... A_row_end]
        atomicAdd(Csub, sum[0]);
    } // bx < nblockrow
}

template <typename Index, typename T>
__global__ void bmm16_sparse_masked_v4(const ushort* __restrict__ A, const ushort* __restrict__ B, T* C,
                                   const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
                                   const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
                                   const Index nblockrows, const Index nblocks, const int nrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {
        GET_LANEID;
        T* Csub = &C[0];
        register T sum[1] = {0};

        // load
        int A_row_start = A_rowptr[bx];
        int A_row_end = A_rowptr[bx+1];
        const ushort* Asub = &(A[A_row_start*16]);

        for (int i=A_row_start; i<(int)ceil(float(A_row_end)/2)*2; i+=2) {
            ushort r0 = 0;

            if (i*16+laneid < A_row_end*16) {
                register unsigned Cm[16] = {0};
                r0 = Asub[(i-A_row_start)*16+laneid];


                int A_col = A_colind[i+laneid/16];
                int B_row_start = B_rowptr[A_col];
                int B_row_end = B_rowptr[A_col+1];
                const ushort* Bsub = &(B[B_row_start*16]);
                for (int j=B_row_start; j<B_row_end; j+=1) {
                    ushort r1 = Bsub[(j-B_row_start)*16+laneid%16];
                    int B_col = B_colind[j];

                    /* bmm */
                    #pragma unroll
                    for (int k=0; k<16; k++)
                    {
                        ushort r2 = __shfl_sync(0xFFFFFFFF, r1, k+(laneid/16)*16); //__shfl(r1, k+(laneid/16)*16);
                        Cm[k] = __popc(r0 & r2);
                    }
                    /* bmm */

                    // load mask
                    Index mask_ind = binarySearch(A_colind, B_col, A_row_start, A_row_end);
                    if (mask_ind != -1) {
                        unsigned mask_val = Asub[(mask_ind-A_row_start)*16+laneid%16];

                        if (mask_val) {
//                            T sum = 0;
                            // add to C_csrval
                            for (int l=0; l<16; l++) {
                                sum[0] += (T)(((mask_val>>(15-l))&0x1)?Cm[l]:0);
                            }
//                            atomicAdd(Csub, sum);
                        }
                    } // mask_ind != -1

                } // j in [B_row_start ... B_row_end]
            } // i*4+laneid < A_row_end*4
        } // i in [A_row_start ... A_row_end]
        atomicAdd(Csub, sum[0]);
    } // bx < nblockrow
}

// traverse from A <-- verified
template <typename Index, typename T>
__global__ void bmm32_sparse_masked_v4(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
                                   const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
                                   const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
                                   const Index nblockrows, const Index nblocks, const int nrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {
        GET_LANEID;
        T* Csub = &C[0];
        register T sum[1] = {0};

        // load
        int A_row_start = A_rowptr[bx];
        int A_row_end = A_rowptr[bx+1];
        const unsigned* Asub = &(A[A_row_start*32]);

        for (int i=A_row_start; i<A_row_end; i++) {
            unsigned r0 = Asub[(i-A_row_start)*32+laneid];

            int A_col = A_colind[i];
            int B_row_start = B_rowptr[A_col];
            int B_row_end = B_rowptr[A_col+1];
            const unsigned* Bsub = &(B[B_row_start*32]);
            for (int j=B_row_start; j<B_row_end; j++) {
                unsigned r1 = Bsub[(j-B_row_start)*32+laneid];
                int B_col = B_colind[j];
                register unsigned Cm[32] = {0};

                /* bmm */
                #pragma unroll
                for (int k=0; k<32; k++)
                {
                    unsigned r2 = __shfl_sync(0xFFFFFFFF, r1, k);
                    Cm[k] = __popc(r0 & r2);
                }
                /* bmm */

                // load mask
                Index mask_ind = binarySearch(A_colind, B_col, A_row_start, A_row_end);
                if (mask_ind != -1) {
                    unsigned mask_val = Asub[(mask_ind-A_row_start)*32+laneid];

                    if (mask_val) {
//                        T sum = 0;
                        // add to C_csrval
                        for (int l=0; l<32; l++) {
                            sum[0] += (T)(((mask_val>>(31-l))&0x1)?Cm[l]:0);
                        }
//                        atomicAdd(Csub, sum);
                    }
                } // mask_ind != -1
            } // j in [B_row_start ... B_row_end]
        } // i in [A_row_start ... A_row_end]
        atomicAdd(Csub, sum[0]);
    } // bx < nblockrow
}

// traverse from C <-- verified
template <typename Index, typename T>
__global__ void bmm32_sparse_masked_v5(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
                                   const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
                                   const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
                                   const Index nblockrows, const Index nblocks, const int nrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {
        GET_LANEID;
        T* Csub = &C[0];
        T sum = 0;

        int A_row_start = A_rowptr[bx];
        int A_row_end = A_rowptr[bx+1];

        for (Index i=A_row_start; i<A_row_end; i++) {
            Index mask_col = A_colind[i];
            unsigned mask_val = A[i*32+laneid];
            register unsigned Cm[32] = {0};

            // collect the result of bmms
            for (Index j=A_row_start; j<A_row_end; j++) {
                Index A_col = A_colind[j];
                Index B_row_start = B_rowptr[A_col];
                Index B_row_end = B_rowptr[A_col+1];

                Index B_col = binarySearch(B_colind, mask_col, B_row_start, B_row_end);

                if (B_col != -1) {
                    unsigned r0 = A[j*32+laneid];
                    unsigned r1 = B[B_col*32+laneid];

                    /* bmm */
                    #pragma unroll
                    for (int k=0; k<32; k++)
                    {
                        unsigned r2 = __shfl_sync(0xFFFFFFFF, r1, k);
                        Cm[k] += (T)(__popc(r0 & r2));
                    }
                    /* bmm */
                }
            }

            // do masking
            for (int l=0; l<32; l++)
            {
                sum += (T)(((mask_val>>(31-l))&0x1)?Cm[l]:0);
            }
        }

        atomicAdd(Csub, sum);
    }
}


//======================================================================================
// New spgemm (with sparse output storage)
//======================================================================================
// set the nnzb in C
template <typename Index, typename T>
__global__ void bmm32_sparse_symbolic(const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
                                      const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
                                      Index* C_rowptr, const Index nblockrows)
{
    // C00 = A00*B00 + A01*B10 + A02*B20 + ... + A0n*Bn0
    Index row_start = 0;
    C_rowptr[0] = 0;
    for(Index C_row=0; C_row<nblockrows; C_row++) {

        // fix C_row to A_row
        Index A_row_start = A_rowptr[C_row];
        Index A_row_end = A_rowptr[C_row+1];

        // for each C_col along the row
        for(Index C_col=0; C_col<nblockrows; C_col++) {
            bool exist = false;

            // check each A's block
            for (Index i=A_row_start; i<A_row_end; i++) {
                Index A_col = A_colind[i];
                Index B_row_start = B_rowptr[A_col];
                Index B_row_end = B_rowptr[A_col+1];

                // check if can be found in B
                Index B_col = binarySearch(B_colind, C_col, B_row_start, B_row_end);
                if (B_col != -1) exist = true;
            }

            if (exist) row_start++; //C_colind[row_start++] = C_col;
        }
        C_rowptr[C_row+1] = row_start; printf("rowptr[%d] = %d\n", C_row+1, row_start);
    }
}


// <-- use with spgemm symbolic, this is correct
// set the nnzb in C
template <typename Index, typename T>
__global__ void bmm32_sparse_numeric(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C_csrval,
                                    const Index* __restrict__ A_rowptr, const Index* __restrict__ A_colind,
                                    const Index* __restrict__ B_rowptr, const Index* __restrict__ B_colind,
                                    const Index* __restrict__ C_rowptr, Index* C_colind, const Index nblockrows)
{

    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    Index C_row = bx;
    if (C_row < nblockrows) {

        GET_LANEID;
        // fix C_row, process 1 row per lane
        Index C_row_start = C_rowptr[C_row*32+laneid];
        Index C_row_end = C_rowptr[C_row*32+laneid+1];

        Index A_row_start = A_rowptr[C_row];
        Index A_row_end = A_rowptr[C_row+1];

        for(Index C_col=0; C_col<nblockrows; C_col++) {

            bool exist = false;
            register T Cm[32] = {0};

            // check each A's block
            for (Index i=A_row_start; i<A_row_end; i++) {
                unsigned r0 = A[i*32+laneid];

                Index A_col = A_colind[i];
                Index B_row_start = B_rowptr[A_col];
                Index B_row_end = B_rowptr[A_col+1];

                // check if can be found in B
                Index B_col = binarySearch(B_colind, C_col, B_row_start, B_row_end);
                if (B_col != -1) {
                    exist = true;

                    unsigned r1 = B[B_col*32+laneid];
                    /* bmm */
                    #pragma unroll
                    for (int k=0; k<32; k++)
                    {
                        unsigned r2 = __shfl_sync(0xFFFFFFFF, r1, k);
                        Cm[k] += (T)(__popc(r0 & r2));
                    }
                    /* bmm */
                }
            } // i

            // update result
            if (exist) {
                Index row = C_row_start;
                for(int k=0; k<32; k++) {
                    if (Cm[k]) {  // set colind & csrval
                        C_colind[row] = C_col*32+k;
                        C_csrval[row] = Cm[k];
                        row++;
                    }
                }
                C_row_start = row;
            } // exist
        } // C_col
    } // bx < nblockrow
}

// currently do not match the baseline
template <typename Index, typename T>
__global__ void reduceSum_masked(const T* __restrict__ C_csrVal, const int nrows, int* gOut,
                                const Index* __restrict__ A_csrRowPtr, const Index* __restrict__ A_csrColInd,
                                const Index* __restrict__ C_csrRowPtr, const Index* __restrict__ C_csrColInd)
{

    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    Index A_row = bx;

    if (A_row < nrows) {
        Index A_row_start = A_csrRowPtr[A_row];
        Index A_row_end = A_csrRowPtr[A_row+1];
        Index C_row_start = C_csrRowPtr[A_row];
        Index C_row_end = C_csrRowPtr[A_row+1];

        for(Index j=A_row_start; j<A_row_end; j++) {
            Index A_col = A_csrColInd[j];
            Index C_col = binarySearch(C_csrColInd, A_col, C_row_start, C_row_end);

            if (C_col != -1) {
                atomicAdd(gOut, (int)C_csrVal[C_col]);

//                printf("j: %d, A_col: %d, C_col:%d, C_csrVal: %d\n", j, A_col, C_col, (int)C_csrVal[C_col]);
//                printf("# %d %d %d\n", j, A_col, (int)C_csrVal[C_col]);
            }

        }
    }
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
