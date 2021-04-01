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

//======================================================================================
// bit-packing
//======================================================================================

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
        unsigned r0 = __brev(__ballot_sync(0xFFFFFFFF, f0>0?1:0)); //__brev(__ballot(f0>0));
        if (laneid == i) Bval = r0;
    }
    B[by*32+laneid] = Bval;
}

// row-major packing bit 32
template <typename T>
__global__ void ToBit32Row(const T* __restrict__ A, unsigned* B, const int A_height, const int A_width, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < nblockrows) {
        unsigned Bval=0;
#pragma unroll
        for (int i=0; i<32; i++)
        {
            T f0 = A[bx*32+i];
            Bval = (Bval<<1) + (f0>0);
        }
        B[bx] = Bval;
    }
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
        T f0 = A[by*64*64+bx*64*32+i*64+laneid];
        T f1 = A[by*64*64+bx*64*32+i*64+32+laneid];
        unsigned r0 = __ballot_sync(0xFFFFFFFF, f0>0?1:0);
        unsigned r1 = __ballot_sync(0xFFFFFFFF, f1>0?1:0);

//        unsigned r0 = __ballot(f0>0);
//        unsigned r1 = __ballot(f1>0);

        ullong l0;
        asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r0),"r"(r1)); //lo,hi
        if (laneid == i) Bval = __brevll(l0);
    }
    B[by*64+bx*32+laneid] = Bval;
}

// row-major packing bit 64
template <typename T>
__global__ void ToBit64Row(const T* __restrict__  A, ullong* B, const int A_height, const int A_width, const int nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < nblockrows) {
        GET_LANEID;

        ullong Bval = 0;
#pragma unroll
        for (int i=0; i<64; i++)
        {
            T f0 = A[bx*64+i];
            Bval = (Bval<<1) | (f0>0);
        }
        B[bx] = Bval;
    }
}

//======================================================================================
// baseline
//======================================================================================
// process consecutive 8 blockrows at a time, result in 32 consecutive rows
template <typename Index, typename T>
__global__ void bmv4_sparse(const uchar* __restrict__ A, const uchar* __restrict__ B, T* C,
                            const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                            const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx*8 <= nblockrows) {
        // load
        GET_LANEID;

        if (bx*8+laneid/4<nblockrows) { //
            int row_start=0, row_end=0, load=0;
            row_start = rowptr[bx*8+laneid/4];
            row_end = rowptr[bx*8+laneid/4+1];
            load = row_end-row_start;

            if (load != 0) {
                const uchar* Asub = &(A[(row_start*4)]); //<--
                const uchar* Bsub = &(B[0]);
                T* Csub = &(C[bx*32]);
                register unsigned Cm[1] = {0};

                // compute 4 blocks on 4 consecutive blockrow at a time
                for(int i=0; i<(int)ceil((float)load/4)*4; i+=4) {
                    uchar a0 = i*4+(laneid%4) < load*4 ? Asub[i*4+(laneid%4)] : 0;
                    uchar a1 = i*4+4+(laneid%4) < load*4 ? Asub[i*4+4+(laneid%4)] : 0;
                    uchar a2 = i*4+8+(laneid%4) < load*4 ? Asub[i*4+8+(laneid%4)] : 0;
                    uchar a3 = i*4+12+(laneid%4) < load*4 ? Asub[i*4+12+(laneid%4)] : 0;
                    unsigned r0 = a0 << 24 | a1 << 16 | a2 << 8 | a3;

                    uchar b0 = i*4+(laneid%4) < load*4 ? Bsub[colind[row_start+i]] : 0;
                    uchar b1 = i*4+4+(laneid%4) < load*4 ? Bsub[colind[row_start+i+1]] : 0;
                    uchar b2 = i*4+8+(laneid%4) < load*4 ? Bsub[colind[row_start+i+2]] : 0;
                    uchar b3 = i*4+12+(laneid%4) < load*4 ? Bsub[colind[row_start+i+3]] : 0;
                    unsigned r1 = b0 << 24 | b1 << 16 | b2 << 8 | b3;

                    Cm[0] += __popc(r0 & r1);
                }

                // store
                Csub[laneid] += (T)(Cm[0]);

            } // if load!=0
        } // <nblockrows
    }
}


template <typename Index, typename T>
__global__ void bmv8_sparse(const uchar* __restrict__ A, const uchar* __restrict__ B, T* C,
                            const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                            const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx*4 <= nblockrows) {
        // load
        GET_LANEID;

        if (bx*4+laneid/8<nblockrows) {
            int row_start=0, row_end=0, load=0;
            row_start = rowptr[bx*4+laneid/8];
            row_end = rowptr[bx*4+laneid/8+1];
            load = row_end-row_start;

            if (load != 0) {
                const uchar* Asub = &(A[row_start*8]);
                const uchar* Bsub = &(B[0]);
                T* Csub = &(C[bx*32]);
                register unsigned Cm[1] = {0};

                // compute 4 blocks on 4 consecutive blockrow at a time
                for(int i=0; i<(int)ceil((float)load/4)*4; i+=4) {
                    uchar a0 = i*8+(laneid%8) < load*8 ? Asub[i*8+(laneid%8)] : 0;
                    uchar a1 = i*8+8+(laneid%8) < load*8 ? Asub[i*8+8+(laneid%8)] : 0;
                    uchar a2 = i*8+16+(laneid%8) < load*8 ? Asub[i*8+16+(laneid%8)] : 0;
                    uchar a3 = i*8+24+(laneid%8) < load*8 ? Asub[i*8+24+(laneid%8)] : 0;
                    unsigned r0 = a0 << 24 | a1 << 16 | a2 << 8 | a3;

                    uchar b0 = i*8+(laneid%8) < load*8 ? Bsub[colind[row_start+i]] : 0;
                    uchar b1 = i*8+8+(laneid%8) < load*8 ? Bsub[colind[row_start+i+1]] : 0;
                    uchar b2 = i*8+16+(laneid%8) < load*8 ? Bsub[colind[row_start+i+2]] : 0;
                    uchar b3 = i*8+24+(laneid%8) < load*8 ? Bsub[colind[row_start+i+3]] : 0;
                    unsigned r1 = b0 << 24 | b1 << 16 | b2 << 8 | b3;

                    Cm[0] += __popc(r0 & r1);
                }

                // store
                Csub[laneid] += (T)(Cm[0]);

            } // if load!=0
        } // <nblockrows
    }
}

template <typename Index, typename T>
__global__ void bmv16_sparse(const ushort* __restrict__ A, const ushort* __restrict__ B, T* C,
                            const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                            const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx*2 <= nblockrows) {
        // load
        GET_LANEID;

        if(bx*2+laneid/16<nblockrows) {
            int row_start=0, row_end=0, load=0;
            row_start = rowptr[bx*2+laneid/16];
            row_end = rowptr[bx*2+laneid/16+1];
            load = row_end-row_start;

            if (load != 0) {
                const ushort* Asub = &(A[row_start*16]);
                const ushort* Bsub = &(B[0]);
                T* Csub = &(C[bx*32]);
                register unsigned Cm[1] = {0};

                // compute 2 blocks on 2 consecutive blockrow at a time
                for(int i=0; i<(int)ceil((float)load/2)*2; i+=2) {
                    ushort a0 = i*16+(laneid%16) < load*16 ? Asub[i*16+(laneid%16)] : 0;
                    ushort a1 = i*16+16+(laneid%16) < load*16 ? Asub[i*16+16+(laneid%16)] : 0;
                    unsigned r0 = a0 << 16 | a1;

                    ushort b0 = i*16+(laneid%16) < load*16 ? Bsub[colind[row_start+i]] : 0;
                    ushort b1 = i*16+16+(laneid%16) < load*16 ? Bsub[colind[row_start+i+1]] : 0;
                    unsigned r1 = b0 << 16 | b1;

                    Cm[0] += __popc(r0 & r1);
                }

                // store
                Csub[laneid] += (T)(Cm[0]);
            } // load != 0
        } // <nblockrows
    }
}

// bsr bmv32 no padding
// A (bsr matrix) * B (vector) = C (vector)
// col-bin(32 x (blocksize x nblocks)) * row-bin((nblockrows x nblocks) x 1) = (nblockrow x nblocks) x 1
template <typename Index, typename T>
__global__ void bmv32_sparse(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
            const Index* __restrict__ rowptr, const Index* __restrict__ colind,
            const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {
        // load
        int row_start = rowptr[bx]; // 0 32 64 . . . 991 [0...nblockrows-1]
        int row_end = rowptr[bx+1]; // 32 64 96 . . . 991 1022 [1...nblockrows]

        if (row_start != row_end) {

            GET_LANEID;
            const unsigned* Asub = &(A[row_start*32]); // block is in continuous layout
            const unsigned* Bsub = &(B[0]); // 0, when it is mv
            T* Csub = &(C[bx*32]);
            register unsigned Cm[1] = {0}; // allocate 1 register

            // compute
            // if that row has more than 1 col block
            for (int i=row_start; i<row_end; i++) {
//                Cm[0] = 0;
                unsigned r0 = Asub[(i-row_start)*32+laneid]; // block is in continuous layout
                unsigned r1 = Bsub[(colind[i])]; // only first row is required

                Cm[0] += __popc(r0 & r1);

            }

             // store
             Csub[laneid] += (T)(Cm[0]); //Csub[laneid] = (T)(Cm[0]>0);
        }
    }
}

// bsr bmv64 no padding
// A (bsr matrix) * B (vector) = C (vector)
template <typename Index, typename T>
__global__ void bmv64_sparse(const ullong* __restrict__ A, const ullong* __restrict__ B, T* C,
                            const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                            const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < nblockrows) {
        // load
        unsigned row_start = rowptr[bx];
        unsigned row_end = rowptr[bx+1];

        if (row_start != row_end) {
            GET_LANEID;
            const ullong* Asub = &(A[row_start*64]);
            const ullong* Bsub = &(B[0]);
            T* Csub = &(C[bx*64]);
            register unsigned Cm[1] = {0};

            // compute
            for (int i=row_start; i<row_end; i++) {
    //            Cm[0] = 0;
                ullong a0 = Asub[(i-row_start)*64+laneid];
                ullong a1 = Asub[(i-row_start)*64+32+laneid];
                ullong b0 = Bsub[colind[i]];

                Cm[0] += (__popcll(a0 & b0) << 16) + __popcll(a1 & b0);
            }

            // store
            short t0, t1;
            asm volatile("mov.b32 {%0,%1}, %2;":"=h"(t1),"=h"(t0):"r"(Cm[0]));
            Csub[laneid] += (T)t0;
            Csub[laneid+32] += (T)t1;
        }
    }
}

//======================================================================================
// multi-shared vector
//======================================================================================
template <typename Index, typename T>
__global__ void bmv8_sparse_sharedvector_multi(const uchar* __restrict__ A, const uchar* __restrict__ B, T* C,
                                               const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                               const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;
    const int split = 2;

    if ((bx/split) <= nblockrows/128) { // bx%split indicate the colGroupSize portion

        // load vector to shared, nblockrows = 7104
        const int nbr = 67561; // <-- this should be predefined
        const int colGroupSize = (nbr+split-1)/split; // 1776
        const int unit = (colGroupSize+1024-1)/1024; //2
        const int sharedMemSize = unit * 1024; //64 * 1024; // 96 * 1024; <-- this should be larger than blockrow
        __shared__ uchar shared_B[sharedMemSize];

        const int maxlimit = min(((bx%split)+1)*colGroupSize, nblockrows);
        #pragma unroll
        for(int i=0; i<unit; i++) {
            shared_B[tid*unit+i] = (bx%split)*colGroupSize + tid*unit+i < maxlimit ? B[(bx%split)*colGroupSize + tid*unit + i] : 0;
        }

        __syncthreads();

        // compute
        // we got 32 warp to process 32 * 4 blockrow,
        // resulting 32*32 rows

        // the below is in a warp
        GET_LANEID;

        if((bx/split)*128+(tid/32)*4+laneid/8<nblockrows) {
            int row_start=0, row_end=0, load=0;
            row_start = rowptr[(bx/split)*128+(tid/32)*4+laneid/8];
            row_end = rowptr[(bx/split)*128+(tid/32)*4+laneid/8+1];
            load = row_end-row_start;

            if (load!= 0) {
                const uchar* Asub = &(A[row_start*8]);
                const uchar* Bsub = &(shared_B[0]);
                const int* colindsub = &(colind[row_start]);
                T* Csub = &(C[(bx/split)*1024]);
                register unsigned Cm[1] = {0};

                register unsigned a[4] = {0};
                register unsigned b[4] = {0};

                // compute 4 blocks on 4 consecutive blockrow at a time
                for(int i=0; i<(int)ceil((float)load/4)*4; i+=4) {
                    a[0] = (i*8+(laneid%8) < load*8 ? Asub[i*8+(laneid%8)] : 0) << 24;
                    a[1] = (i*8+8+(laneid%8) < load*8 ? Asub[i*8+8+(laneid%8)] : 0) << 16;
                    a[2] = (i*8+16+(laneid%8) < load*8 ? Asub[i*8+16+(laneid%8)] : 0) << 8;
                    a[3] = (i*8+24+(laneid%8) < load*8 ? Asub[i*8+24+(laneid%8)] : 0);

                    b[0] = ((i*8+(laneid%8) < load*8 && ((bx%split) == (colindsub[i]/colGroupSize))) ? Bsub[(colindsub[i]%colGroupSize)] : 0) << 24;
                    b[1] = ((i*8+8+(laneid%8) < load*8 && ((bx%split) == (colindsub[i+1]/colGroupSize))) ? Bsub[(colindsub[i+1]%colGroupSize)] : 0) << 16;
                    b[2] = ((i*8+16+(laneid%8) < load*8 && ((bx%split) == (colindsub[i+2]/colGroupSize))) ? Bsub[(colindsub[i+2]%colGroupSize)] : 0) << 8;
                    b[3] = ((i*8+24+(laneid%8) < load*8 && ((bx%split) == (colindsub[i+3]/colGroupSize))) ? Bsub[(colindsub[i+3]%colGroupSize)] : 0);

                    Cm[0] += __popc((a[0]|a[1]|a[2]|a[3]) & (b[0]|b[1]|b[2]|b[3]));
                }

                // store
                atomicAdd(Csub+tid, (T)(Cm[0]));
            } // load != 0
        } // tid... <nblockrows
    } // (bx/4)*128 <= nblockrows
}

template <typename Index, typename T>
__global__ void bmv8_sparse_sharedallunsigned_multi(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
                                        const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                        const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if ((bx/4) <= nblockrows/128) { // bx%4 indicate the colGroupSize portion
//        if (tid == 0) printf("row: %d, part: %d\n", bx/4, bx%4);

        // load vector to shared, nblockrows = 7104
        const int nbr = 35623; // <-- this should be predefined
        const int colGroupSize = (nbr+4-1)/4; // 1776
        const int unit = (colGroupSize+1024-1)/1024; //2
        const int sharedMemSize = unit * 1024; //64 * 1024; // 96 * 1024; <-- this should be larger than blockrow
        __shared__ unsigned shared_B[sharedMemSize];

        const int maxlimit = min(((bx%4)+1)*colGroupSize, nblockrows);
        #pragma unroll
        for(int i=0; i<unit; i++) {
            shared_B[tid*unit+i] = (bx%4)*colGroupSize + tid*unit+i < maxlimit ? B[(bx%4)*colGroupSize + tid*unit + i] : 0;
        }

        __syncthreads();

        // compute
        // we got 32 warp to process 32 * 4 blockrow,
        // resulting 32*32 rows

        // the below is in a warp
        GET_LANEID;

        if((bx/4)*128+(tid/32)*4+laneid/8<nblockrows) {
            int row_start=0, row_end=0, load=0;
            row_start = rowptr[(bx/4)*128+(tid/32)*4+laneid/8];
            row_end = rowptr[(bx/4)*128+(tid/32)*4+laneid/8+1];
            load = row_end-row_start; // <- become 4 aligned

            if (load != 0) {
                const unsigned* Asub = &(A[(row_start/4)*8]);
                const unsigned* Bsub = &(shared_B[0]);
                const int* colindsub = &(colind[row_start]);
                T* Csub = &(C[(bx/4)*1024]);
                register unsigned Cm[1] = {0};
                register unsigned b[4] = {0};

                // compute 4 blocks on 4 consecutive blockrow at a time
                for(int i=0; i<load; i+=4) {
                    unsigned r0 = Asub[(i/4)*8+(laneid%8)];

                    b[0] = ((bx%4) == ((colindsub[i]/4)/colGroupSize)) ? ((((0xFF000000 >> ((colindsub[i]%4)*8)) & Bsub[(colindsub[i]/4)%colGroupSize]) >> (24 - ((colindsub[i]%4)*8))) << 24) : 0;
                    b[1] = ((bx%4) == ((colindsub[i+1]/4)/colGroupSize)) ? (((0xFF000000 >> (((colindsub[i+1]%4)*8)) & Bsub[(colindsub[i+1]/4)%colGroupSize]) >> (24 - ((colindsub[i+1]%4)*8))) << 16) : 0;
                    b[2] = ((bx%4) == ((colindsub[i+2]/4)/colGroupSize)) ? ((((0xFF000000 >> ((colindsub[i+2]%4)*8)) & Bsub[(colindsub[i+2]/4)%colGroupSize]) >> (24 - ((colindsub[i+2]%4)*8))) << 8) : 0;
                    b[3] = ((bx%4) == ((colindsub[i+3]/4)/colGroupSize)) ? ((((0xFF000000 >> ((colindsub[i+3]%4)*8)) & Bsub[(colindsub[i+3]/4)%colGroupSize]) >> (24 - ((colindsub[i+3]%4)*8)))) : 0;

                    Cm[0] += __popc(r0 & (b[0]|b[1]|b[2]|b[3]));
                }

                // store
                atomicAdd(Csub+tid, (T)(Cm[0]));
            } // load != 0
        } // tid... <nblockrows
    } // bx*128 <= nblockrows
}

//======================================================================================
// bfs.cu
//======================================================================================
__global__ void printOneBin8Vec (const uchar packvec)
{
    uchar j;
    for(j = 1 << 7; j > 0; j = j / 2)
        (packvec & j) ? printf("1") : printf("0");
    printf(" ");
}

__global__ void printOneBin32Vec (const unsigned packvec)
{
    unsigned j;
    for(j = 1 << 31; j > 0; j = j / 2)
        (packvec & j) ? printf("1") : printf("0");
    printf(" ");
}

// 32 thread in a warp
template <typename Index, typename T>
__global__ void bmv4_sparse_bin_masked_v4(const uchar* __restrict__ A, const uchar* __restrict__ B, uchar* C,
                                        const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                        const Index nblockrows, const uchar* __restrict__ mask)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*8 <= nblockrows) {
        // compute

        // load
        GET_LANEID;
        int row = bx*8+laneid/4;

        if (row<nblockrows) {

            int row_start, row_end, load=0;
            row_start = rowptr[row];
            row_end = rowptr[row+1];
            load = row_end-row_start;

            if (load != 0) {
                const uchar* Asub = &(A[row_start*4]);
                const uchar* Bsub = &(B[0]);
                const Index* colindsub = &(colind[row_start]);
                uchar* Csub = &(C[bx*8]);
                register unsigned Cm[1] = {0};
//                register unsigned a[4] = {0};
//                register unsigned b[4] = {0};

                #pragma unroll
                for(int i=0; i<load; i+=1) { //(((load+4-1)/4)*4)
                    unsigned r0 = Asub[i*4+laneid%4];
                    unsigned r1 = Bsub[(colindsub[i])];
                    Cm[0] += __popc(r0 & r1);

//                    a[0] = (i*4+(laneid%4) < load*4 ? Asub[i*4+(laneid%4)] : 0) << 24;
//                    a[1] = (i*4+4+(laneid%4) < load*4 ? Asub[i*4+4+(laneid%4)] : 0) << 16;
//                    a[2] = (i*4+8+(laneid%4) < load*4 ? Asub[i*4+8+(laneid%4)] : 0) << 8;
//                    a[3] = (i*4+12+(laneid%4) < load*4 ? Asub[i*4+12+(laneid%4)] : 0);
//
//                    b[0] = (i*4+(laneid%4) < load*4 ? Bsub[colind[row_start+i]] : 0) << 24;
//                    b[1] = (i*4+4+(laneid%4) < load*4 ? Bsub[colind[row_start+i+1]] : 0) << 16;
//                    b[2] = (i*4+8+(laneid%4) < load*4 ? Bsub[colind[row_start+i+2]] : 0) << 8;
//                    b[3] = (i*4+12+(laneid%4) < load*4 ? Bsub[colind[row_start+i+3]] : 0);
//
//
//                    Cm[0] += __popc((a[0]|a[1]|a[2]|a[3]) & (b[0]|b[1]|b[2]|b[3]));
                }

                // store
                unsigned r2 = __ballot_sync(0xFFFFFFFF, Cm[0]>0?1:0);
                uchar temp = (uchar)((((__brev(r2) >> (28-((laneid/4)*4))) & 0xF) & (~mask[row]))& 0x0F);

//                printf("%d, %u\n", (laneid/4), temp);
                Csub[(laneid/4)] = temp;
            }
        }
    }
}

// result in binary
template <typename Index, typename T>
__global__ void bmv32_sparse_bin_masked_v2(const unsigned* __restrict__ A, const unsigned* __restrict__ B, unsigned* C,
                                        const Index* __restrict__ rowptr, const Index* __restrict__ colind, 
                                        const Index nblockrows, const unsigned* __restrict__ mask)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < nblockrows) {
        // load
        int row_start = rowptr[bx];
        int row_end = rowptr[bx+1];

        if (row_start != row_end) {

            GET_LANEID;
            const unsigned warpid = (threadIdx.x >> 5);

            const unsigned* Asub = &(A[row_start*32]);
            const unsigned* Bsub = &(B[0]);
            unsigned* Csub = &(C[bx]);
            register unsigned Cm[1] = {0};

            // compute
            for (int i=row_start; i<row_end; i++) {

                unsigned r0 = Asub[(i-row_start)*32+laneid];
                unsigned r1 = Bsub[(colind[i])];

                Cm[0] += __popc(r0 & r1);

            }

             // store
             unsigned r2 = __ballot_sync(0xFFFFFFFF, (T)Cm[0]>0?1:0);
             Csub[warpid] = (__brev(r2) & (~mask[bx+warpid]));
        }
    }
}

// 1024 trd per warp
template <typename Index, typename T>
__global__ void bmv32_sparse_bin_masked_v4(const unsigned* __restrict__ A, const unsigned* __restrict__ B, unsigned* C,
                                        const Index* __restrict__ rowptr, const Index* __restrict__ colind, 
                                        const Index nblockrows, const unsigned* __restrict__ mask)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*32 <= nblockrows) {
        // compute
        // we got 32 warp to process 32 * 1 blockrow,
        // resulting 32*32 rows

        // load
        GET_LANEID;
        const unsigned warpid = (tid >> 5);

        if (bx*32+tid/32<nblockrows) {

            int row_start, row_end, load=0;
            row_start = rowptr[bx*32+tid/32];
            row_end = rowptr[bx*32+tid/32+1];
            load = row_end-row_start;

            if (load != 0) {
                const unsigned* Asub = &(A[row_start*32]);
                const unsigned* Bsub = &(B[0]);
                const int* colindsub = &(colind[row_start]);
                unsigned* Csub = &(C[bx*32]);
                register unsigned Cm[1] = {0};

                #pragma unroll
                for(int i=0; i<load; i++) {
                    unsigned r0 = Asub[i*32+laneid];
                    unsigned r1 = Bsub[(colindsub[i])];
                    Cm[0] += __popc(r0 & r1);
                }

                // store
                unsigned r2 = __ballot_sync(0xFFFFFFFF, (T)Cm[0]>0?1:0);
                Csub[warpid] = (__brev(r2) & (~mask[bx*32+warpid]));
            }
        }
    }
}

// test new thread model
template <typename Index, typename T>
__global__ void bmv32_sparse_bin_masked_v1(const unsigned* __restrict__ A, const unsigned* __restrict__ B, unsigned* C,
                                        const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                        const Index nblockrows, const unsigned* __restrict__ mask)
{
    Index row = blockIdx.x * blockDim.x + threadIdx.x;

    for (; row < nblockrows; row += gridDim.x*blockDim.x) {

        unsigned discoverable = 0; // to hold temporary bmv result
        unsigned val = mask[row];

        if (val == 0xFFFFFFFF) {
        } else {
            Index row_start = rowptr[row];
            Index row_end = rowptr[row+1];

            for(; row_start < row_end; row_start++) {
                Index col_ind = colind[row_start];
                val = mask[col_ind];

                if (val != 0xFFFFFFFF) {
                    unsigned r0;
                    unsigned r1 = B[col_ind];
                    unsigned Cm = 0;

                    for(int i=0; i<32; i++) {
                        r0 = A[row_start*32+i];
                        Cm = Cm << 1 | (__popc(r0 & r1) > 0);
                    }

                    // early exit
                    discoverable |= Cm;
                    if (discoverable == 0xFFFFFFFF) break;
                }
            }
            C[row] = (discoverable & (~mask[row]));
        } 
    }
}

// test 32 thrd per warp
template <typename Index, typename T>
__global__ void bmv32_sparse_bin_masked_v3(const unsigned* __restrict__ A, const unsigned* __restrict__ B, unsigned* C,
                                        const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                        const Index nblockrows, const unsigned* __restrict__ mask)
{
    const unsigned row = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (row < nblockrows) {

        unsigned val = mask[row];

        if (val == 0xFFFFFFFF) {
        } else {
            Index row_start = rowptr[row];
            Index row_end = rowptr[row+1];
            GET_LANEID;
            unsigned Cm[1] = {0};

            for(; row_start < row_end; row_start++) {
                Index col_ind = colind[row_start];
                val = mask[col_ind];

                if (val != 0xFFFFFFFF) {
                    unsigned r0;
                    unsigned r1 = B[col_ind];

                    r0 = A[row_start*32+laneid];
                    Cm[0] += __popc(r0 & r1);

                    // // early exit
                    // discoverable |= Cm;
                    // if (discoverable == 0xFFFFFFFF) break;
                }
            }

            // store
            unsigned r2 = __ballot_sync(0xFFFFFFFF, (T)(Cm[0])>0?1:0);
            C[row] = (__brev(r2) & (~mask[row]));
        } 
    }
}

//1024 thrd per tb
template <typename Index, typename T>
__global__ void bmv32_sparse_bin_masked_v5(const unsigned* __restrict__ A, const unsigned* __restrict__ B, unsigned* C,
                                        const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                        const Index nblockrows, const unsigned* __restrict__ mask)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*32 <= nblockrows) {
        // compute
        // we got 32 warp to process 32 * 1 blockrow,
        // resulting 32*32 rows

        // load
        unsigned row = bx*32+tid/32;

        if (row<nblockrows) {
           unsigned val = mask[row];
           

            if (val == 0xFFFFFFFF) {
            } else {
                Index row_start = rowptr[row];
                Index row_end = rowptr[row+1];

                GET_LANEID;
                const unsigned warpid = (tid >> 5);

                unsigned Cm[1] = {0};
                for(; row_start < row_end; row_start++) {
                    Index col_ind = colind[row_start];
                    val = mask[col_ind];

                    if (val != 0xFFFFFFFF) {
                        unsigned r0;
                        unsigned r1 = B[col_ind];

                        r0 = A[row_start*32+laneid];
                        Cm[0] += __popc(r0 & r1);

                        // // early exit
                        // discoverable |= Cm;
                        // if (discoverable == 0xFFFFFFFF) break;
                    }
                }

                // store
                unsigned r2 = __ballot_sync(0xFFFFFFFF, (T)(Cm[0])>0?1:0);
                C[row] = (__brev(r2) & (~mask[row]));
            }

        }

    }
}


__global__ void reduce_naive(const unsigned* __restrict__ vec, const int N, int* succptr) {
    int count = 0;
    for(int i=0; i<N; i++) {
        count += __popc(vec[i]);
    }
    *succptr = count;
    // printf("count: %d\n", count);
}

template <typename T>
__global__ void reduce(const T* __restrict__ vec, const int N, int* succptr) {

    if(blockIdx.x*1024+threadIdx.x < N) atomicAdd(succptr, __popc((unsigned)vec[blockIdx.x*1024+threadIdx.x]));
}

template <typename T>
__global__ void OR(T* vec1, const int N, const T* __restrict__ vec2) {
    if (blockIdx.x*1024+threadIdx.x < N) vec1[blockIdx.x*1024+threadIdx.x] |= vec2[blockIdx.x*1024+threadIdx.x];
}

template <typename T>
__global__ void fillZero(T* vec, const int N) {
    if (blockIdx.x*1024+threadIdx.x < N) vec[blockIdx.x*1024+threadIdx.x] = 0;
}

__global__ void resetSuccptr(int* succptr) {
    succptr[0] = 0;
}

template <typename T>
__global__ void Mask(T* vec1, const int N, const T* __restrict__ vec2) {
    for(int i=0; i<N; i++) {
        vec1[i] &= (~vec2[i]);
    }
}

//======================================================================================
// sssp.cu
//======================================================================================
__global__ void fillMax(unsigned* vec, const int N, const unsigned maxval) {
    if (blockIdx.x*1024+threadIdx.x < N) vec[blockIdx.x*1024+threadIdx.x] = maxval;
}

__global__ void setSource(unsigned* vec, const int s) {
    vec[s] = 0;
}

__global__ void resetSuccptrUnsigned(unsigned* succptr) {
    succptr[0] = 0.f;
}

__global__ void fillValUnsigned(unsigned* vec, const int N, const unsigned val) {
    if (blockIdx.x*1024+threadIdx.x < N) vec[blockIdx.x*1024+threadIdx.x] = val;
}

__global__ void reduceAddUnsigned(unsigned* resptr, const int N, const unsigned* vec) {
    if (blockIdx.x*1024+threadIdx.x < N) atomicAdd(resptr, (vec[blockIdx.x*1024+threadIdx.x] != 2147483647) ? vec[blockIdx.x*1024+threadIdx.x] : 0);
}


template <typename Index, typename T>
__global__ void bmv4_sparse_full_minplus(const uchar* __restrict__ A, const T* __restrict__ B, T* C,
                                         const Index* __restrict__ rowptr, const Index* __restrict__ colind, const Index nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*8 <= nblockrows) {
        // compute

        // load
        GET_LANEID;
        int row = bx*8+laneid/4;

        if (row<nblockrows) {
            int row_start, row_end, load=0;
            row_start = rowptr[row];
            row_end = rowptr[row+1];
            load = row_end-row_start;

            if (load != 0) {
                const uchar* Asub = &(A[row_start*4]);
                const T* Bsub = &(B[0]);
                const Index* colindsub = &(colind[row_start]);
                T* Csub = &(C[bx*32]);
                T minval = 2147483647;
                register T f[4] = {0};
                register Index jind[4] = {0};
                register uchar a[4] = {0};

                #pragma unroll
                for(int i=0; i<(((load+4-1)/4)*4); i+=4) {
                    a[0] = i*4+(laneid%4) < load*4 ? Asub[i*4+(laneid%4)] : 0;
                    a[1] = i*4+4+(laneid%4) < load*4 ? Asub[i*4+4+(laneid%4)] : 0;
                    a[2] = i*4+8+(laneid%4) < load*4 ? Asub[i*4+8+(laneid%4)] : 0;
                    a[3] = i*4+12+(laneid%4) < load*4 ? Asub[i*4+12+(laneid%4)] : 0;

                    for(int j=0; j<4; j++) {
                        jind[0] = i*4+(laneid%4) < load*4 ? colindsub[i]*4+j : 0;
                        jind[1] = i*4+4+(laneid%4) < load*4 ? colindsub[i+1]*4+j : 0;
                        jind[2] = i*4+8+(laneid%4) < load*4 ? colindsub[i+2]*4+j : 0;
                        jind[3] = i*4+12+(laneid%4) < load*4 ? colindsub[i+3]*4+j : 0;

                        f[0] = i*4+(laneid%4) < load*4 ? Bsub[jind[0]] : 0;
                        f[1] = i*4+4+(laneid%4) < load*4 ? Bsub[jind[1]] : 0;
                        f[2] = i*4+8+(laneid%4) < load*4 ? Bsub[jind[2]] : 0;
                        f[3] = i*4+12+(laneid%4) < load*4 ? Bsub[jind[3]] : 0;

                        /* relax the binary matrix */
                        for(int l=0; l<4; l++) {
                            T m;
                            if (((a[l]>>(3-j))&0x1) == 0) {
                                if (jind[l] == bx*32+laneid) m = 0;
                                else m = 2147483647;
                            } else {
                                if (jind[l] == bx*32+laneid) m = 0;
                                else m = 1;
                            }
                            T res = (f[l] == 2147483647 || m == 2147483647) ? 2147483647 : (f[l]+m);
                            minval = min(res, minval);
                        }
                    }
                }

//                #pragma unroll
//                for(int i=0; i<load; i+=1) {
//                    a[0] = Asub[i*4+(laneid%4)];
//
//                    for(int j=0; j<4; j++) {
//                        jind[0] = colindsub[i]*4+j;
//                        f[0] = Bsub[jind[0]];
//
//                        /* relax the binary matrix */
//                        T m;
//                        if (((a[0]>>(3-j))&0x1) == 0) {
//                            if (jind[0] == bx*32+laneid) m = 0;
//                            else m = 2147483647;
//                        } else {
//                            if (jind[0] == bx*32+laneid) m = 0;
//                            else m = 1;
//                        }
//                        T res = (f[0] == 2147483647 || m == 2147483647) ? 2147483647 : (f[0]+m);
//                        minval = min(res, minval);
//                    }
//                }

                // store
                Csub[laneid] = minval;
            }
        }
    }
}

// 512 trd per warp
template <typename Index, typename T>
__global__ void bmv32_sparse_full_minplus(const unsigned* __restrict__ A, const T* __restrict__ B, T* C,
                                        const Index* __restrict__ rowptr, const Index* __restrict__ colind, const Index nblockrows)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*16 <= nblockrows) {

        // load
        GET_LANEID;
        const unsigned warpid = (tid >> 5);
        if (bx*16+tid/32<nblockrows) {
            int row_start, row_end, load=0;
            row_start = rowptr[bx*16+tid/32];
            row_end = rowptr[bx*16+tid/32+1];
            load = row_end-row_start;

            if (load != 0) {
                const unsigned* Asub = &(A[row_start*32]);
                const T* Bsub = &(B[0]);
                const int* colindsub = &(colind[row_start]);
                T* Csub = &(C[bx*512]);
                T minval = 2147483647;

                #pragma unroll
                for(int i=0; i<load; i++) {
                    unsigned r0 = Asub[i*32+laneid];

                    #pragma unroll
                    for(int j=0; j<32; j++) {
                        int b_ind = (colindsub[i])*32+j;
                        T f1 = Bsub[b_ind]; // vector

                        /* relax the binary matrix */
                        T f2; // matrix
                        if (((r0>>(31-j)) & 0x1) == 0) {
                            if (b_ind == bx*512+warpid*32+laneid) f2 = 0;
                            else f2 = 2147483647;
                        } else {
                            if (b_ind == bx*512+warpid*32+laneid) f2 = 0;
                            else f2 = 1;
                        }

                        T res = (f1 == 2147483647 || f2 == 2147483647) ? 2147483647 : (f1+f2);
                        minval = min(res, minval);
                    }
                }

                // store
                Csub[warpid*32+laneid] = minval;
            }
        }

    }

}

__global__ void ewiseLess(unsigned* resvec, const int N, const unsigned* vec1, const unsigned* vec2) {
    if (blockIdx.x*1024+threadIdx.x < N) resvec[blockIdx.x*1024+threadIdx.x] = (unsigned)(vec1[blockIdx.x*1024+threadIdx.x] < vec2[blockIdx.x*1024+threadIdx.x]);
}

__global__ void ewiseMin(unsigned* resvec, const int N, const unsigned* vec1, const unsigned* vec2) {
    if (blockIdx.x*1024+threadIdx.x < N) resvec[blockIdx.x*1024+threadIdx.x] = min(vec1[blockIdx.x*1024+threadIdx.x], vec2[blockIdx.x*1024+threadIdx.x]);
}

__global__ void assignMax(unsigned* resvec, const int N, const unsigned* mask, const unsigned maxval) {
    if (blockIdx.x*1024+threadIdx.x < N) resvec[blockIdx.x*1024+threadIdx.x] = (mask[blockIdx.x*1024+threadIdx.x] == 0) ? maxval : resvec[blockIdx.x*1024+threadIdx.x];
}

//======================================================================================
// pr.cu
//======================================================================================
template <typename Index, typename T>
__global__ void bmv4_sparse_full(const uchar* __restrict__ A, const T* __restrict__ B, T* C,
                                  const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                  const Index nblockrows, const Index* __restrict__ csrrowptr, const float alpha)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*8 <= nblockrows) {
        // compute

        // load
        GET_LANEID;
        int row = bx*8+laneid/4;

        if (row<nblockrows) {
            int row_start, row_end, load=0;
            row_start = rowptr[row];
            row_end = rowptr[row+1];
            load = row_end-row_start;

            if (load != 0) {
                const uchar* Asub = &(A[row_start*4]);
                const T* Bsub = &(B[0]);
                const Index* colindsub = &(colind[row_start]);
                T* Csub = &(C[bx*32]);
                T sum = 0;
                register T f[4] = {0};
                register Index nnz[4] = {1};

                #pragma unroll
                for(int i=0; i<(((load+4-1)/4)*4); i+=4) {
                    uchar a0 = i*4+(laneid%4) < load*4 ? Asub[i*4+(laneid%4)] : 0;
                    uchar a1 = i*4+4+(laneid%4) < load*4 ? Asub[i*4+4+(laneid%4)] : 0;
                    uchar a2 = i*4+8+(laneid%4) < load*4 ? Asub[i*4+8+(laneid%4)] : 0;
                    uchar a3 = i*4+12+(laneid%4) < load*4 ? Asub[i*4+12+(laneid%4)] : 0;

                    for(int j=0; j<4; j++) {
                        Index j0 = i*4+(laneid%4) < load*4 ? colindsub[i]*4+j : 0;
                        Index j1 = i*4+4+(laneid%4) < load*4 ? colindsub[i+1]*4+j : 0;
                        Index j2 = i*4+8+(laneid%4) < load*4 ? colindsub[i+2]*4+j : 0;
                        Index j3 = i*4+12+(laneid%4) < load*4 ? colindsub[i+3]*4+j : 0;

                        f[0] = i*4+(laneid%4) < load*4 ? Bsub[j0] : 0;
                        f[1] = i*4+4+(laneid%4) < load*4 ? Bsub[j1] : 0;
                        f[2] = i*4+8+(laneid%4) < load*4 ? Bsub[j2] : 0;
                        f[3] = i*4+12+(laneid%4) < load*4 ? Bsub[j3] : 0;

                        nnz[0] = i*4+(laneid%4) < load*4 ? (csrrowptr[j0+1]-csrrowptr[j0]) :0;
                        nnz[1] = i*4+4+(laneid%4) < load*4 ? (csrrowptr[j1+1]-csrrowptr[j1]) :0;
                        nnz[2] = i*4+8+(laneid%4) < load*4 ? (csrrowptr[j2+1]-csrrowptr[j2]) :0;
                        nnz[3] = i*4+12+(laneid%4) < load*4 ? (csrrowptr[j3+1]-csrrowptr[j3]) :0;

                        sum +=(((a0>>(3-j))&0x1)?(alpha*f[0]/nnz[0]):0);
                        sum +=(((a1>>(3-j))&0x1)?(alpha*f[1]/nnz[1]):0);
                        sum +=(((a2>>(3-j))&0x1)?(alpha*f[2]/nnz[2]):0);
                        sum +=(((a3>>(3-j))&0x1)?(alpha*f[3]/nnz[3]):0);
                    }
                }

//                #pragma unroll
//                for(int i=0; i<load; i+=1) {
//                    uchar a0 = Asub[i*4+(laneid%4)];
//
//                    for(int j=0; j<4; j++) {
//                        Index j0 = colindsub[i]*4+j;
//                        f[0] = Bsub[j0];
//                        nnz[0] = csrrowptr[j0+1]-csrrowptr[j0];
//
//                        sum +=(((a0>>(3-j))&0x1)?(alpha*f[0]/nnz[0]):0);
//                    }
//                }

                // store
                Csub[laneid] = sum;
            }
        }
    }
}


// bin (A) * full (B) = full (C)
// 512 trd per warp
template <typename Index, typename T>
__global__ void bmv32_sparse_full(const unsigned* __restrict__ A, const T* __restrict__ B, T* C,
                                  const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                  const Index nblockrows, const Index* csrrowptr, const float alpha)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*16 <= nblockrows) {
        // compute

        // load
        GET_LANEID;
        const unsigned warpid = (tid >> 5);

        if (bx*16+tid/32<nblockrows) {

            int row_start, row_end, load=0;
            row_start = rowptr[bx*16+tid/32];
            row_end = rowptr[bx*16+tid/32+1];
            load = row_end-row_start;

            if (load != 0) {
                const unsigned* Asub = &(A[row_start*32]);
                const T* Bsub = &(B[0]);
                const int* colindsub = &(colind[row_start]);
                T* Csub = &(C[bx*512]);
                T sum = 0;

                #pragma unroll
                for(int i=0; i<load; i++) {
                    unsigned r0 = Asub[i*32+laneid];

                    #pragma unroll
                    for(int j=0; j<32; j++) {
                        Index jind = (colindsub[i])*32+j;
                        T f1 = Bsub[jind];
                        Index nnz = (csrrowptr[jind+1]-csrrowptr[jind]);

                        sum +=(((r0>>(31-j))&0x1)?(alpha*f1/nnz):0);
                    }
                }

                // store
                Csub[warpid*32+laneid] = sum;
            }
        }
    }
}

template <typename Index, typename T>
__global__ void bmv32_sparse_full_singlewarp(const unsigned* __restrict__ A, const T* __restrict__ B, T* C,
                                            const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                            const Index nblockrows, const Index* __restrict__ csrrowptr, const float alpha)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx < nblockrows) {
        // load
        GET_LANEID;

        int row_start, row_end, load=0;
        row_start = rowptr[bx];
        row_end = rowptr[bx+1];
        load = row_end-row_start;

        if (load != 0) {
            const unsigned* Asub = &(A[row_start*32]);
            const T* Bsub = &(B[0]);
            const int* colindsub = &(colind[row_start]);
            T* Csub = &(C[bx*32]);
            T sum = 0;

            #pragma unroll
            for(int i=0; i<load; i++) {
                unsigned r0 = Asub[i*32+laneid];


                #pragma unroll
                for(int j=0; j<32; j++) {
                    Index jind = (colindsub[i])*32+j;
                    T f1 = Bsub[jind];
                    Index nnz = (csrrowptr[jind+1]-csrrowptr[jind]);

                    sum +=(((r0>>(31-j))&0x1)?(alpha*f1/nnz):0);

                }
            }

            // store
            Csub[bx*32+laneid] = sum;
        }
    }
}

__global__ void fillVal(float* vec, const int N, const float val) {
    if (blockIdx.x*1024+threadIdx.x < N) vec[blockIdx.x*1024+threadIdx.x] = val;
}

// resvec = (vec + val)
__global__ void ewiseAddVal(float* resvec, const int N, const float* vec, const float val) {
    if (blockIdx.x*1024+threadIdx.x < N) resvec[blockIdx.x*1024+threadIdx.x] = vec[blockIdx.x*1024+threadIdx.x] + val;
}

// resvec = (vec1-vec2)
__global__ void ewiseSubVec(float* resvec, const int N, const float* vec1, const float* vec2) {
    if (blockIdx.x*1024+threadIdx.x < N) resvec[blockIdx.x*1024+threadIdx.x] = (vec1[blockIdx.x*1024+threadIdx.x] - vec2[blockIdx.x*1024+threadIdx.x]);
}

// resvec = (vec1*vec2)
__global__ void ewiseMul(float* resvec, const int N, const float* vec1, const float* vec2) {
    if (blockIdx.x*1024+threadIdx.x < N) resvec[blockIdx.x*1024+threadIdx.x] = (vec1[blockIdx.x*1024+threadIdx.x] * vec2[blockIdx.x*1024+threadIdx.x]);
}

__global__ void reduceAdd(float* resptr, const int N, const float* vec) {
    if (blockIdx.x*1024+threadIdx.x < N) atomicAdd(resptr, vec[blockIdx.x*1024+threadIdx.x]);
}


__global__ void resetErrorptr(float* errorptr) {
    errorptr[0] = 0;
}

//======================================================================================
// cc.cu
//======================================================================================
//// resvec = min(vec1, vec2)
//__global__ void ewiseMin(float* resvec, const int N, const float* vec1, const float* vec2) {
//    if (blockIdx.x*1024+threadIdx.x < N) resvec[blockIdx.x*1024+threadIdx.x] = min(vec1[blockIdx.x*1024+threadIdx.x], vec2[blockIdx.x*1024+threadIdx.x]);
//}

// f[f[u]] = mngf[u]
__global__ void assignScatter(float* resvec, const int N, const float* vec1, const float* vec2) {
    if (blockIdx.x*1024+threadIdx.x < N) resvec[(int)(vec2[blockIdx.x*1024+threadIdx.x])] = vec1[blockIdx.x*1024+threadIdx.x];
}

// gf[u] = f[f[u]]
__global__ void extractGather(float* resvec, const int N, const float* vec) {
    if (blockIdx.x*1024+threadIdx.x < N) resvec[blockIdx.x*1024+threadIdx.x] = vec[(int)(vec[blockIdx.x*1024+threadIdx.x])];
}

// resvec = vec1 != vec2
__global__ void ewiseNotEqual(float* resvec, const int N, const float* vec1, const float* vec2) {
    if (blockIdx.x*1024+threadIdx.x < N) resvec[blockIdx.x*1024+threadIdx.x] = (float)(vec1[blockIdx.x*1024+threadIdx.x] != vec2[blockIdx.x*1024+threadIdx.x]);
}

//// assign
//__global__ void assignMax(float* resvec, const int N, const float* vec, const float val) {
//    if (blockIdx.x*1024+threadIdx.x < N) resvec[blockIdx.x*1024+threadIdx.x] = vec[blockIdx.x*1024+threadIdx.x] ? val : 0;
//}


//======================================================================================
// new model -- more warps in a thread block
//======================================================================================
// 1024 threads/thread block <-- maximum
// vector load into shared memory to shared across sm
template <typename Index, typename T>
__global__ void bmv8_sparse_sharedvector(const uchar* __restrict__ A, const uchar* __restrict__ B, T* C,
                                        const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                        const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*128 <= nblockrows) {

        // load vector to shared
        const int unit = 28;//48;
        const int sharedMemSize = unit * 1024; //64 * 1024; // 96 * 1024; <-- this should be larger than blockrow
        __shared__ uchar shared_B[sharedMemSize];

        #pragma unroll
        for(int i=0; i<unit; i++) {
            shared_B[tid*unit+i] = tid*unit+i < nblockrows ? B[tid*unit+i] : 0;
        }

        __syncthreads();

        // compute
        // we got 32 warp to process 32 * 4 blockrow,
        // resulting 32*32 rows

        // the below is in a warp
        GET_LANEID;

        if(bx*128+(tid/32)*4+laneid/8<nblockrows) {
            int row_start=0, row_end=0, load=0;
            row_start = rowptr[bx*128+(tid/32)*4+laneid/8];
            row_end = rowptr[bx*128+(tid/32)*4+laneid/8+1];
            load = row_end-row_start;

            if (load!= 0) {
                const uchar* Asub = &(A[row_start*8]);
                const uchar* Bsub = &(shared_B[0]);
                T* Csub = &(C[bx*1024]);
                register unsigned Cm[1] = {0};

                // compute 4 blocks on 4 consecutive blockrow at a time
                for(int i=0; i<(int)ceil((float)load/4)*4; i+=4) {
                    uchar a0 = i*8+(laneid%8) < load*8 ? Asub[i*8+(laneid%8)] : 0;
                    uchar a1 = i*8+8+(laneid%8) < load*8 ? Asub[i*8+8+(laneid%8)] : 0;
                    uchar a2 = i*8+16+(laneid%8) < load*8 ? Asub[i*8+16+(laneid%8)] : 0;
                    uchar a3 = i*8+24+(laneid%8) < load*8 ? Asub[i*8+24+(laneid%8)] : 0;
                    unsigned r0 = a0 << 24 | a1 << 16 | a2 << 8 | a3;

                    uchar b0 = i*8+(laneid%8) < load*8 ? Bsub[colind[row_start+i]] : 0;
                    uchar b1 = i*8+8+(laneid%8) < load*8 ? Bsub[colind[row_start+i+1]] : 0;
                    uchar b2 = i*8+16+(laneid%8) < load*8 ? Bsub[colind[row_start+i+2]] : 0;
                    uchar b3 = i*8+24+(laneid%8) < load*8 ? Bsub[colind[row_start+i+3]] : 0;
                    unsigned r1 = b0 << 24 | b1 << 16 | b2 << 8 | b3;

                    Cm[0] += __popc(r0 & r1);
                }

                // store
                Csub[tid] += (T)(Cm[0]);
            } // load != 0
        } // tid... <nblockrows
    } // bx*128 <= nblockrows
}

// for better code readability, still use % * /
// but it can be replace with & << >>
template <typename Index, typename T>
__global__ void bmv8_sparse_sharedvectorunsigned(const uchar* __restrict__ A, const unsigned* __restrict__ B, T* C,
                                        const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                        const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*128 <= nblockrows) {

        // load vector to shared
        const int unit = 7;
        const int sharedMemSize = unit * 1024; //64 * 1024; // 96 * 1024; <-- this should be larger than blockrow
        __shared__ unsigned shared_B[sharedMemSize];

        #pragma unroll
        for(int i=0; i<unit; i++) {
            shared_B[tid*unit+i] = tid*unit+i < nblockrows ? B[tid*unit+i] : 0;
        }

        __syncthreads();

        // compute
        // we got 32 warp to process 32 * 4 blockrow,
        // resulting 32*32 rows

        // the below is in a warp
        GET_LANEID;

        if(bx*128+(tid/32)*4+laneid/8<nblockrows) {
            int row_start=0, row_end=0, load=0;
            row_start = rowptr[bx*128+(tid/32)*4+laneid/8];
            row_end = rowptr[bx*128+(tid/32)*4+laneid/8+1];
            load = row_end-row_start;

            if (load!= 0) {
                const uchar* Asub = &(A[row_start*8]);
                const unsigned* Bsub = &(shared_B[0]);
                const int* colindsub = &(colind[row_start]);
                T* Csub = &(C[bx*1024]);
                register unsigned Cm[1] = {0};

                // compute 4 blocks on 4 consecutive blockrow at a time
                for(int i=0; i<(int)ceil((float)load/4)*4; i+=4) {
                    uchar a0 = i*8+(laneid%8) < load*8 ? Asub[i*8+(laneid%8)] : 0;
                    uchar a1 = i*8+8+(laneid%8) < load*8 ? Asub[i*8+8+(laneid%8)] : 0;
                    uchar a2 = i*8+16+(laneid%8) < load*8 ? Asub[i*8+16+(laneid%8)] : 0;
                    uchar a3 = i*8+24+(laneid%8) < load*8 ? Asub[i*8+24+(laneid%8)] : 0;
                    unsigned r0 = a0 << 24 | a1 << 16 | a2 << 8 | a3;

                    uchar b0 = i*8+(laneid%8) < load*8 ? (((0xFF000000 >> ((colindsub[i]%4)*8)) & Bsub[colindsub[i]/4]) >> (24 - ((colindsub[i]%4)*8))) : 0;
                    uchar b1 = i*8+8+(laneid%8) < load*8 ? (((0xFF000000 >> ((colindsub[i+1]%4)*8)) & Bsub[colindsub[i+1]/4]) >> (24 - ((colindsub[i+1]%4)*8))) : 0;
                    uchar b2 = i*8+16+(laneid%8) < load*8 ? (((0xFF000000 >> ((colindsub[i+2]%4)*8)) & Bsub[colindsub[i+2]/4]) >> (24 - ((colindsub[i+2]%4)*8))) : 0;
                    uchar b3 = i*8+24+(laneid%8) < load*8 ? (((0xFF000000 >> ((colindsub[i+3]%4)*8)) & Bsub[colindsub[i+3]/4]) >> (24 - ((colindsub[i+3]%4)*8))) : 0;
                    unsigned r1 = b0 << 24 | b1 << 16 | b2 << 8 | b3;

                    Cm[0] += __popc(r0 & r1);
                }

                // store
                Csub[tid] += (T)(Cm[0]);
            } // load != 0
        } // tid... <nblockrows
    } // bx*128 <= nblockrows
}

// remove uchar, put in register <-- now using
template <typename Index, typename T>
__global__ void bmv8_sparse_sharedallunsigned(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
                                        const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                        const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*128 <= nblockrows) {

        // load vector to shared
        const int unit = 7;
        const int sharedMemSize = unit * 1024; //64 * 1024; // 96 * 1024; <-- this should be larger than blockrow
        __shared__ unsigned shared_B[sharedMemSize];

        #pragma unroll
        for(int i=0; i<unit; i++) {
            shared_B[tid*unit+i] = tid*unit+i < nblockrows ? B[tid*unit+i] : 0;
        }

        __syncthreads();

        // compute
        // we got 32 warp to process 32 * 4 blockrow,
        // resulting 32*32 rows

        // the below is in a warp
        GET_LANEID;

        if(bx*128+(tid/32)*4+laneid/8<nblockrows) {
            int row_start=0, row_end=0, load=0;
            row_start = rowptr[bx*128+(tid/32)*4+laneid/8];
            row_end = rowptr[bx*128+(tid/32)*4+laneid/8+1];
            load = row_end-row_start; // <- become 4 aligned

            if (load != 0) {
                const unsigned* Asub = &(A[(row_start/4)*8]);
                const unsigned* Bsub = &(shared_B[0]);
                const int* colindsub = &(colind[row_start]);
                T* Csub = &(C[bx*1024]);
                register unsigned Cm[1] = {0};
                register unsigned b[4] = {0};

                // compute 4 blocks on 4 consecutive blockrow at a time
                for(int i=0; i<load; i+=4) {
                    unsigned r0 = Asub[(i/4)*8+(laneid%8)];

                    b[0] = (((0xFF000000 >> ((colindsub[i]%4)*8)) & Bsub[colindsub[i]/4]) >> (24 - ((colindsub[i]%4)*8))) << 24;
                    b[1] = (((0xFF000000 >> ((colindsub[i+1]%4)*8)) & Bsub[colindsub[i+1]/4]) >> (24 - ((colindsub[i+1]%4)*8))) << 16;
                    b[2] = (((0xFF000000 >> ((colindsub[i+2]%4)*8)) & Bsub[colindsub[i+2]/4]) >> (24 - ((colindsub[i+2]%4)*8))) << 8;
                    b[3] = (((0xFF000000 >> ((colindsub[i+3]%4)*8)) & Bsub[colindsub[i+3]/4]) >> (24 - ((colindsub[i+3]%4)*8)));

                    Cm[0] += __popc(r0 & (b[0]|b[1]|b[2]|b[3]));
                }

                // store
                Csub[tid] += (T)(Cm[0]);
            } // load != 0
        } // tid... <nblockrows
    } // bx*128 <= nblockrows
}


// 0.128
template <typename Index, typename T>
__global__ void bmv8_sparse_allunsigned(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
                                        const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                        const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*128 <= nblockrows) {
        // compute
        // we got 32 warp to process 32 * 4 blockrow,
        // resulting 32*32 rows

        // the below is in a warp
        GET_LANEID;

        if(bx*128+(tid/32)*4+laneid/8<nblockrows) {
            int row_start=0, row_end=0, load=0;
            row_start = rowptr[bx*128+(tid/32)*4+laneid/8];
            row_end = rowptr[bx*128+(tid/32)*4+laneid/8+1];
            load = row_end-row_start; // <- become 4 aligned

            if (load != 0) {
                const unsigned* Asub = &(A[(row_start/4)*8]);
                const unsigned* Bsub = &(B[0]);
                const int* colindsub = &(colind[row_start]);
                T* Csub = &(C[bx*1024]);
                register unsigned Cm[1] = {0};
                register unsigned b[4] = {0};

                // compute 4 blocks on 4 consecutive blockrow at a time
                for(int i=0; i<load; i+=4) {
                    unsigned r0 = Asub[(i/4)*8+(laneid%8)];

                    b[0] = (((0xFF000000 >> ((colindsub[i]%4)*8)) & Bsub[colindsub[i]/4]) >> (24 - ((colindsub[i]%4)*8))) << 24;
                    b[1] = (((0xFF000000 >> ((colindsub[i+1]%4)*8)) & Bsub[colindsub[i+1]/4]) >> (24 - ((colindsub[i+1]%4)*8))) << 16;
                    b[2] = (((0xFF000000 >> ((colindsub[i+2]%4)*8)) & Bsub[colindsub[i+2]/4]) >> (24 - ((colindsub[i+2]%4)*8))) << 8;
                    b[3] = (((0xFF000000 >> ((colindsub[i+3]%4)*8)) & Bsub[colindsub[i+3]/4]) >> (24 - ((colindsub[i+3]%4)*8)));

                    Cm[0] += __popc(r0 & (b[0]|b[1]|b[2]|b[3]));
                }

                // store
                Csub[tid] += (T)(Cm[0]);
            } // load != 0
        } // tid... <nblockrows
    } // bx*128 <= nblockrows
}

// baseline new, remove uchar, 1024 threads
// 0.125 (no share, no unsigned)
template <typename Index, typename T>
__global__ void bmv8_sparse_new(const uchar* __restrict__ A, const uchar* __restrict__ B, T* C,
                                        const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                        const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*128 <= nblockrows) {
        // compute
        // we got 32 warp to process 32 * 4 blockrow,
        // resulting 32*32 rows

        // the below is in a warp
        GET_LANEID;

        if(bx*128+(tid/32)*4+laneid/8<nblockrows) {
            int row_start=0, row_end=0, load=0;
            row_start = rowptr[bx*128+(tid/32)*4+laneid/8];
            row_end = rowptr[bx*128+(tid/32)*4+laneid/8+1];
            load = row_end-row_start;

            if (load!= 0) {
                const uchar* Asub = &(A[row_start*8]);
                const uchar* Bsub = &(B[0]);
                T* Csub = &(C[bx*1024]);
                register unsigned Cm[1] = {0};
                register unsigned a[4] = {0};
                register unsigned b[4] = {0};

                // compute 4 blocks on 4 consecutive blockrow at a time
                for(int i=0; i<(int)ceil((float)load/4)*4; i+=4) {
                    a[0] = (i*8+(laneid%8) < load*8 ? Asub[i*8+(laneid%8)] : 0) << 24;
                    a[1] = (i*8+8+(laneid%8) < load*8 ? Asub[i*8+8+(laneid%8)] : 0) << 16;
                    a[2] = (i*8+16+(laneid%8) < load*8 ? Asub[i*8+16+(laneid%8)] : 0) << 8;
                    a[3] = (i*8+24+(laneid%8) < load*8 ? Asub[i*8+24+(laneid%8)] : 0);

                    b[0] = (i*8+(laneid%8) < load*8 ? Bsub[colind[row_start+i]] : 0) << 24;
                    b[1] = (i*8+8+(laneid%8) < load*8 ? Bsub[colind[row_start+i+1]] : 0) << 16;
                    b[2] = (i*8+16+(laneid%8) < load*8 ? Bsub[colind[row_start+i+2]] : 0) << 8;
                    b[3] = (i*8+24+(laneid%8) < load*8 ? Bsub[colind[row_start+i+3]] : 0);


                    Cm[0] += __popc((a[0]|a[1]|a[2]|a[3]) & (b[0]|b[1]|b[2]|b[3]));
                }

                // store
                Csub[tid] += (T)(Cm[0]);
            } // load != 0
        } // tid... <nblockrows
    } // bx*128 <= nblockrows
}

// baseline new, remove uchar, 1024 threads
// 0.130 (no share, vector unsigned)
template <typename Index, typename T>
__global__ void bmv8_sparse_new2(const uchar* __restrict__ A, const unsigned* __restrict__ B, T* C,
                                        const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                        const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*128 <= nblockrows) {
        // compute
        // we got 32 warp to process 32 * 4 blockrow,
        // resulting 32*32 rows

        // the below is in a warp
        GET_LANEID;

        if(bx*128+(tid/32)*4+laneid/8<nblockrows) {
            int row_start=0, row_end=0, load=0;
            row_start = rowptr[bx*128+(tid/32)*4+laneid/8];
            row_end = rowptr[bx*128+(tid/32)*4+laneid/8+1];
            load = row_end-row_start;

            if (load != 0) {
                const uchar* Asub = &(A[row_start*8]);
                const unsigned* Bsub = &(B[0]);
                const int* colindsub = &(colind[row_start]);
                T* Csub = &(C[bx*1024]);

                register unsigned Cm[1] = {0};
                register unsigned a[4] = {0};
                register unsigned b[4] = {0};

                // compute 4 blocks on 4 consecutive blockrow at a time
                for(int i=0; i<(int)ceil((float)load/4)*4; i+=4) {
                    a[0] = (i*8+(laneid%8) < load*8 ? Asub[i*8+(laneid%8)] : 0) << 24;
                    a[1] = (i*8+8+(laneid%8) < load*8 ? Asub[i*8+8+(laneid%8)] : 0) << 16;
                    a[2] = (i*8+16+(laneid%8) < load*8 ? Asub[i*8+16+(laneid%8)] : 0) << 8;
                    a[3] = (i*8+24+(laneid%8) < load*8 ? Asub[i*8+24+(laneid%8)] : 0);

                    b[0] = (((0xFF000000 >> ((colindsub[i]%4)*8)) & Bsub[colindsub[i]/4]) >> (24 - ((colindsub[i]%4)*8))) << 24;
                    b[1] = (((0xFF000000 >> ((colindsub[i+1]%4)*8)) & Bsub[colindsub[i+1]/4]) >> (24 - ((colindsub[i+1]%4)*8))) << 16;
                    b[2] = (((0xFF000000 >> ((colindsub[i+2]%4)*8)) & Bsub[colindsub[i+2]/4]) >> (24 - ((colindsub[i+2]%4)*8))) << 8;
                    b[3] = (((0xFF000000 >> ((colindsub[i+3]%4)*8)) & Bsub[colindsub[i+3]/4]) >> (24 - ((colindsub[i+3]%4)*8)));



                    Cm[0] += __popc((a[0]|a[1]|a[2]|a[3]) & (b[0]|b[1]|b[2]|b[3]));
                }

                // store
                Csub[tid] += (T)(Cm[0]);
            } // load != 0
        } // tid... <nblockrows
    } // bx*128 <= nblockrows
}


template <typename Index, typename T>
__global__ void bmv8_sparse_twowarp(const uchar* __restrict__ A, const uchar* __restrict__ B, T* C,
                                    const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                    const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x; // 0-64

    if (bx*8 <= nblockrows) {

        // compute
        // we got 2 warp to process 2 * 4 blockrow,
        // resulting 2*32 rows

        // the below is in a warp
        GET_LANEID;

        if(bx*8+(tid/32)*4+laneid/8<nblockrows) {
            int row_start=0, row_end=0, load=0;
            row_start = rowptr[bx*8+(tid/32)*4+laneid/8];
            row_end = rowptr[bx*8+(tid/32)*4+laneid/8+1];
            load = row_end-row_start;

            if (load != 0) {
                const uchar* Asub = &(A[row_start*8]);
                const uchar* Bsub = &(B[0]);
                T* Csub = &(C[bx*64]);
                register unsigned Cm[1] = {0};

                // compute 4 blocks on 4 consecutive blockrow at a time
                for(int i=0; i<(int)ceil((float)load/4)*4; i+=4) {
                    uchar a0 = i*8+(laneid%8) < load*8 ? Asub[i*8+(laneid%8)] : 0;
                    uchar a1 = i*8+8+(laneid%8) < load*8 ? Asub[i*8+8+(laneid%8)] : 0;
                    uchar a2 = i*8+16+(laneid%8) < load*8 ? Asub[i*8+16+(laneid%8)] : 0;
                    uchar a3 = i*8+24+(laneid%8) < load*8 ? Asub[i*8+24+(laneid%8)] : 0;
                    unsigned r0 = a0 << 24 | a1 << 16 | a2 << 8 | a3;

                    uchar b0 = i*8+(laneid%8) < load*8 ? Bsub[colind[row_start+i]] : 0;
                    uchar b1 = i*8+8+(laneid%8) < load*8 ? Bsub[colind[row_start+i+1]] : 0;
                    uchar b2 = i*8+16+(laneid%8) < load*8 ? Bsub[colind[row_start+i+2]] : 0;
                    uchar b3 = i*8+24+(laneid%8) < load*8 ? Bsub[colind[row_start+i+3]] : 0;
                    unsigned r1 = b0 << 24 | b1 << 16 | b2 << 8 | b3;

                    Cm[0] += __popc(r0 & r1);
                }

                // store
                Csub[tid] += (T)(Cm[0]);
            } // load != 0
        } // tid... <nblockrows
    } // bx*8 <= nblockrows
}

template <typename Index, typename T>
__global__ void bmv16_sparse_sharedvector(const ushort* __restrict__ A, const ushort* __restrict__ B, T* C,
                                        const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                        const Index nblockrows, const Index nblocks)
{

    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*64 <= nblockrows) {
        // load vector to shared
        const int unit = 24;
        const int sharedMemSize = unit * 1024; //32 * 1024; //48 * 1024; <-- this should be larger than blockrow
        __shared__ ushort shared_B[sharedMemSize];

        #pragma unroll
        for(int i=0; i<unit; i++) {
            shared_B[tid*unit+i] = tid*unit+i < nblockrows ? B[tid*unit+i] : 0;
        }

        __syncthreads();

        // compute
        // we got 32 warp to process 32 * 2 blockrow,
        // resulting 32*32 rows

        // load
        GET_LANEID;

        if(bx*64+(tid/32)*2+laneid/16<nblockrows) {
            int row_start=0, row_end=0, load=0;
            row_start = rowptr[bx*64+(tid/32)*2+laneid/16];
            row_end = rowptr[bx*64+(tid/32)*2+laneid/16+1];
            load = row_end-row_start;

            if (load != 0) {
                const ushort* Asub = &(A[row_start*16]);
                const ushort* Bsub = &(shared_B[0]);
                T* Csub = &(C[bx*1024]);
                register unsigned Cm[1] = {0};

                // compute 2 blocks on 2 consecutive blockrow at a time
                for(int i=0; i<(int)ceil((float)load/2)*2; i+=2) {
                    ushort a0 = i*16+(laneid%16) < load*16 ? Asub[i*16+(laneid%16)] : 0;
                    ushort a1 = i*16+16+(laneid%16) < load*16 ? Asub[i*16+16+(laneid%16)] : 0;
                    unsigned r0 = a0 << 16 | a1;

                    ushort b0 = i*16+(laneid%16) < load*16 ? Bsub[colind[row_start+i]] : 0;
                    ushort b1 = i*16+16+(laneid%16) < load*16 ? Bsub[colind[row_start+i+1]] : 0;
                    unsigned r1 = b0 << 16 | b1;

                    Cm[0] += __popc(r0 & r1);
                }

                // store
                Csub[tid] += (T)(Cm[0]);

            } // load != 0
        } // tid... <nblockrows
    } // bx*64 <= nblockrows

}

template <typename Index, typename T>
__global__ void bmv16_sparse_sharedallunsigned(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
                                        const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                        const Index nblockrows, const Index nblocks)
{

    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*64 <= nblockrows) {
        // load vector to shared
        const int unit = 7;
        const int sharedMemSize = unit * 1024; //32 * 1024; //48 * 1024; <-- this should be larger than blockrow
        __shared__ unsigned shared_B[sharedMemSize];

        #pragma unroll
        for(int i=0; i<unit; i++) {
            shared_B[tid*unit+i] = tid*unit+i < nblockrows ? B[tid*unit+i] : 0;
        }

        __syncthreads();

        // compute
        // we got 32 warp to process 32 * 2 blockrow,
        // resulting 32*32 rows

        // load
        GET_LANEID;

        if(bx*64+(tid/32)*2+laneid/16<nblockrows) {
            int row_start=0, row_end=0, load=0;
            row_start = rowptr[bx*64+(tid/32)*2+laneid/16];
            row_end = rowptr[bx*64+(tid/32)*2+laneid/16+1];
            load = row_end-row_start;

            if (load != 0) {
                const unsigned* Asub = &(A[(row_start/2)*16]);
                const unsigned* Bsub = &(shared_B[0]);
                const int* colindsub = &(colind[row_start]);
                T* Csub = &(C[bx*1024]);
                register unsigned Cm[1] = {0};
                register unsigned b[2] = {0};

                // compute 2 blocks on 2 consecutive blockrow at a time
                for(int i=0; i<load; i+=2) {
                    unsigned r0 = Asub[(i/2)*16+(laneid%16)];

                    b[0] = (((0xFFFF0000 >> ((colindsub[i]%2)*16)) & Bsub[colindsub[i]/2]) >> (16 - ((colindsub[i]%2)*16))) << 16;
                    b[1] = (((0xFFFF0000 >> ((colindsub[i+1]%2)*16)) & Bsub[colindsub[i+1]/2]) >> (16 - ((colindsub[i+1]%2)*16)));

                    Cm[0] += __popc(r0 & (b[0]|b[1]));
                }

                // store
                Csub[tid] += (T)(Cm[0]);

            } // load != 0
        } // tid... <nblockrows
    } // bx*64 <= nblockrows

}

template <typename Index, typename T>
__global__ void bmv16_sparse_allunsigned(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
                                        const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                        const Index nblockrows, const Index nblocks)
{

    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*64 <= nblockrows) {
        // compute
        // we got 32 warp to process 32 * 2 blockrow,
        // resulting 32*32 rows

        // load
        GET_LANEID;

        if(bx*64+(tid/32)*2+laneid/16<nblockrows) {
            int row_start=0, row_end=0, load=0;
            row_start = rowptr[bx*64+(tid/32)*2+laneid/16];
            row_end = rowptr[bx*64+(tid/32)*2+laneid/16+1];
            load = row_end-row_start;

            if (load != 0) {
                const unsigned* Asub = &(A[(row_start/2)*16]);
                const unsigned* Bsub = &(B[0]);
                const int* colindsub = &(colind[row_start]);
                T* Csub = &(C[bx*1024]);
                register unsigned Cm[1] = {0};
                register unsigned b[2] = {0};

                // compute 2 blocks on 2 consecutive blockrow at a time
                for(int i=0; i<load; i+=2) {
                    unsigned r0 = Asub[(i/2)*16+(laneid%16)];

                    b[0] = (((0xFFFF0000 >> ((colindsub[i]%2)*16)) & Bsub[colindsub[i]/2]) >> (16 - ((colindsub[i]%2)*16))) << 16;
                    b[1] = (((0xFFFF0000 >> ((colindsub[i+1]%2)*16)) & Bsub[colindsub[i+1]/2]) >> (16 - ((colindsub[i+1]%2)*16)));

                    Cm[0] += __popc(r0 & (b[0]|b[1]));
                }

                // store
                Csub[tid] += (T)(Cm[0]);

            } // load != 0
        } // tid... <nblockrows
    } // bx*64 <= nblockrows

}

// bmv16 new
// remove ushort, no share, no unsigned, 1024 threads
template <typename Index, typename T>
__global__ void bmv16_sparse_new(const ushort* __restrict__ A, const ushort* __restrict__ B, T* C,
                                const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                const Index nblockrows, const Index nblocks)
{

    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*64 <= nblockrows) {
        // compute
        // we got 32 warp to process 32 * 2 blockrow,
        // resulting 32*32 rows

        // load
        GET_LANEID;

        if(bx*64+(tid/32)*2+laneid/16<nblockrows) {
            int row_start=0, row_end=0, load=0;
            row_start = rowptr[bx*64+(tid/32)*2+laneid/16];
            row_end = rowptr[bx*64+(tid/32)*2+laneid/16+1];
            load = row_end-row_start;

            if (load != 0) {
                const ushort* Asub = &(A[row_start*16]);
                const ushort* Bsub = &(B[0]);
                const int* colindsub = &(colind[row_start]);
                T* Csub = &(C[bx*1024]);
                register unsigned Cm[1] = {0};
                register unsigned a[2] = {0};
                register unsigned b[2] = {0};

                // compute 2 blocks on 2 consecutive blockrow at a time
                for(int i=0; i<(int)ceil((float)load/2)*2; i+=2) {
                    a[0] = (i*16+(laneid%16) < load*16 ? Asub[i*16+(laneid%16)] : 0) << 16;
                    a[1] = (i*16+16+(laneid%16) < load*16 ? Asub[i*16+16+(laneid%16)] : 0);

                    b[0] = (i*16+(laneid%16) < load*16 ? Bsub[colind[row_start+i]] : 0) << 16;
                    b[1] = (i*16+16+(laneid%16) < load*16 ? Bsub[colind[row_start+i+1]] : 0);

                    Cm[0] += __popc((a[0]|a[1]) & (b[0]|b[1]));
                }

                // store
                Csub[tid] += (T)(Cm[0]);

            } // load != 0
        } // tid... <nblockrows
    } // bx*64 <= nblockrows

}

template <typename Index, typename T>
__global__ void bmv16_sparse_twowarp(const ushort* __restrict__ A, const ushort* __restrict__ B, T* C,
                                    const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                                    const Index nblockrows, const Index nblocks)
{

    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*4 <= nblockrows) {
        // compute
        // we got 2 warp to process 2 * 2 blockrow,
        // resulting 2*32 rows

        // load
        GET_LANEID;

        if(bx*4+(tid/32)*2+laneid/16<nblockrows) {
            int row_start=0, row_end=0, load=0;
            row_start = rowptr[bx*4+(tid/32)*2+laneid/16];
            row_end = rowptr[bx*4+(tid/32)*2+laneid/16+1];
            load = row_end-row_start;

            if (load != 0) {
                const ushort* Asub = &(A[row_start*16]);
                const ushort* Bsub = &(B[0]);
                T* Csub = &(C[bx*64]);
                register unsigned Cm[1] = {0};

                // compute 2 blocks on 2 consecutive blockrow at a time
                for(int i=0; i<(int)ceil((float)load/2)*2; i+=2) {
                    ushort a0 = i*16+(laneid%16) < load*16 ? Asub[i*16+(laneid%16)] : 0;
                    ushort a1 = i*16+16+(laneid%16) < load*16 ? Asub[i*16+16+(laneid%16)] : 0;
                    unsigned r0 = a0 << 16 | a1;

                    ushort b0 = i*16+(laneid%16) < load*16 ? Bsub[colind[row_start+i]] : 0;
                    ushort b1 = i*16+16+(laneid%16) < load*16 ? Bsub[colind[row_start+i+1]] : 0;
                    unsigned r1 = b0 << 16 | b1;

                    Cm[0] += __popc(r0 & r1);
                }

                // store
                Csub[tid] += (T)(Cm[0]);

            } // load != 0
        } // tid... <nblockrows
    } // bx*4 <= nblockrows

}

// bsrbmv32 share, 1024 threads
template <typename Index, typename T>
__global__ void bmv32_sparse_sharedvector(const unsigned* __restrict__ A, const unsigned* __restrict__ B,
            T* C, const int A_height, const int A_width, const int B_width,
            const Index* __restrict__ rowptr, const Index* __restrict__ colind,
            const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*32 <= nblockrows) {
        // load vector to shared
        const int unit = 7;
        const int sharedMemSize = unit * 1024; //32 * 1024; //48 * 1024; <-- this should be larger than blockrow
        __shared__ unsigned shared_B[sharedMemSize];

        #pragma unroll
        for(int i=0; i<unit; i++) {
            shared_B[tid*unit+i] = tid*unit+i < nblockrows ? B[tid*unit+i] : 0;
        }

        __syncthreads();

         // compute
        // we got 32 warp to process 32 * 1 blockrow,
        // resulting 32*32 rows

        // load
        GET_LANEID;

        if (bx*32+tid/32<nblockrows) {
            int row_start, row_end, load=0;
            row_start = rowptr[bx*32+tid/32];
            row_end = rowptr[bx*32+tid/32+1];
            load = row_end-row_start;

            if (load != 0) {
                const unsigned* Asub = &(A[row_start*32]);
                const unsigned* Bsub = &(shared_B[0]);
                const int* colindsub = &(colind[row_start]);
                T* Csub = &(C[bx*1024]);
                register unsigned Cm[1] = {0};

                #pragma unroll
                for(int i=0; i<load; i++) {
                    unsigned r0 = Asub[i*32+laneid];
                    unsigned r1 = Bsub[(colindsub[i])];
                    Cm[0] += __popc(r0 & r1);
                }

                // store
                Csub[tid] += (T)(Cm[0]);
            }
        }
    }
}

// 1024 per tb, no share
template <typename Index, typename T>
__global__ void bmv32_sparse_new(const unsigned* __restrict__ A, const unsigned* __restrict__ B,
            T* C, const int A_height, const int A_width, const int B_width,
            const Index* __restrict__ rowptr, const Index* __restrict__ colind,
            const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*32 <= nblockrows) {
        // compute
        // we got 32 warp to process 32 * 1 blockrow,
        // resulting 32*32 rows

        // load
        GET_LANEID;

        if (bx*32+tid/32<nblockrows) {
            int row_start, row_end, load=0;
            row_start = rowptr[bx*32+tid/32];
            row_end = rowptr[bx*32+tid/32+1];
            load = row_end-row_start;

            if (load != 0) {
                const unsigned* Asub = &(A[row_start*32]);
                const unsigned* Bsub = &(B[0]);
                const int* colindsub = &(colind[row_start]);
                T* Csub = &(C[bx*1024]);
                register unsigned Cm[1] = {0};

                #pragma unroll
                for(int i=0; i<load; i++) {
                    unsigned r0 = Asub[i*32+laneid];
                    unsigned r1 = Bsub[(colindsub[i])];
                    Cm[0] += __popc(r0 & r1);
                }

                // store
                Csub[tid] += (T)(Cm[0]);
            }
        }
    }
}

// bsrbmv64 share, 1024 threads
template <typename Index, typename T>
__global__ void bmv64_sparse_sharedvector(const ullong* __restrict__ A, const ullong* __restrict__ B,
                            T* C, const int A_height, const int A_width, const int B_width,
                            const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                            const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*32 <= nblockrows) {

        // load vector to shared
        const int unit = 4;//48;
        const int sharedMemSize = unit * 1024; //64 * 1024; // 96 * 1024; <-- this should be larger than blockrow
        __shared__ ullong shared_B[sharedMemSize];

        #pragma unroll
        for(int i=0; i<unit; i++) {
            shared_B[tid*unit+i] = tid*unit+i < nblockrows ? B[tid*unit+i] : 0;
        }

        __syncthreads();

        // compute
        // we got 32 warp to process 32 * 1 blockrow,
        // resulting 32*32 rows

        // load
        GET_LANEID;

        if (bx*32+tid/32<nblockrows) {
            int row_start, row_end, load=0;
            row_start = rowptr[bx*32+tid/32];
            row_end = rowptr[bx*32+tid/32+1];
            load = row_end-row_start;

            if (load != 0) {

                const ullong* Asub = &(A[row_start*64]);
                const ullong* Bsub = &(shared_B[0]);
                const int* colindsub = &(colind[row_start]);
                T* Csub = &(C[bx*2048]);
                register unsigned Cm[1] = {0};

                #pragma unroll
                for(int i=0; i<load; i++) {
                    ullong a0 = Asub[i*64+laneid];
                    ullong a1 = Asub[i*64+32+laneid];
                    ullong b0 = Bsub[colindsub[i]];

                    Cm[0] += (__popcll(a0 & b0) << 16) + __popcll(a1 & b0);
                }

                // store
                short t0, t1;
                asm volatile("mov.b32 {%0,%1}, %2;":"=h"(t1),"=h"(t0):"r"(Cm[0]));
                Csub[(tid/32)*64+laneid] += (T)t0;
                Csub[(tid/32)*64+32+laneid] += (T)t1;
            }
        }
    }
}

// 1024 per tb, no share
template <typename Index, typename T>
__global__ void bmv64_sparse_new(const ullong* __restrict__ A, const ullong* __restrict__ B,
                            T* C, const int A_height, const int A_width, const int B_width,
                            const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                            const Index nblockrows, const Index nblocks)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    const unsigned tid = threadIdx.x;

    if (bx*32 <= nblockrows) {

        // compute
        // we got 32 warp to process 32 * 1 blockrow,
        // resulting 32*32 rows

        // load
        GET_LANEID;

        if (bx*32+tid/32<nblockrows) {
            int row_start, row_end, load=0;
            row_start = rowptr[bx*32+tid/32];
            row_end = rowptr[bx*32+tid/32+1];
            load = row_end-row_start;

            if (load != 0) {

                const ullong* Asub = &(A[row_start*64]);
                const ullong* Bsub = &(B[0]);
                const int* colindsub = &(colind[row_start]);
                T* Csub = &(C[bx*2048]);
                register unsigned Cm[1] = {0};

                #pragma unroll
                for(int i=0; i<load; i++) {
                    ullong a0 = Asub[i*64+laneid];
                    ullong a1 = Asub[i*64+32+laneid];
                    ullong b0 = Bsub[colindsub[i]];

                    Cm[0] += (__popcll(a0 & b0) << 16) + __popcll(a1 & b0);
                }

                // store
                short t0, t1;
                asm volatile("mov.b32 {%0,%1}, %2;":"=h"(t1),"=h"(t0):"r"(Cm[0]));
                Csub[(tid/32)*64+laneid] += (T)t0;
                Csub[(tid/32)*64+32+laneid] += (T)t1;
            }
        }
    }
}


//======================================================================================
// Considering Workload Balancing -- workload split, workload merge (not in use)
//======================================================================================
template <typename Index, typename T>
__global__ void bmv32_sparse_workloadsplit(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
                                           const Index* __restrict__ colind, const Index* __restrict__ workloadptr,
                                           const Index workloadsize, const Index MAX, int* runtime, int* load)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < workloadsize) {
#ifdef PROF
        clock_t start_time = clock();
#endif

    // load
    int row = workloadptr[bx*3];
    int row_start = workloadptr[bx*3+1];
    int workload = workloadptr[bx*3+2];

    // compute
    GET_LANEID;
    const unsigned* Asub = &(A[row_start*32]); // block is in continuous layout
    const unsigned* Bsub = &(B[0]); // 0, when it is mv
    T* Csub = &(C[row*32]);
    register unsigned Cm[1] = {0}; // allocate 1 register <-- can parallelize

    // compute
    for (int i=row_start; i<row_start+workload; i++) {

        unsigned r0 = Asub[(i-row_start)*32+laneid]; // block is in continuous layout
        unsigned r1 = Bsub[(colind[i])]; // only first row is required

        Cm[0] += __popc(r0 & r1);
    }

    // store
    atomicAdd(Csub+laneid, (T)(Cm[0])); // <-- can tag the continuety

#ifdef PROF
        clock_t stop_time = clock();
        runtime[bx] = (int)(stop_time - start_time);
        load[bx] = workload;
//        GET_LANEID;
//        if (laneid == 1 && load[bx] == 0) {printf("[%d] %d %d\n", bx, (int)(stop_time - start_time), (int)(row_end-row_start));}
#endif
    }
}

template <typename Index, typename T>
__global__ void bmv32_sparse_workloadsplit_register(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
                                           const Index* __restrict__ colind, const Index* __restrict__ workloadptr,
                                           const Index workloadsize, const Index MAX, int* runtime, int* load)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;
    if (bx < workloadsize) {
#ifdef PROF
        clock_t start_time = clock();
#endif

    // load
    int row = workloadptr[bx*3];
    int row_start = workloadptr[bx*3+1];
    int workload = workloadptr[bx*3+2];

    // compute
    GET_LANEID;
    const unsigned* Asub = &(A[row_start*32]); // block is in continuous layout
    const unsigned* Bsub = &(B[0]); // 0, when it is mv
    T* Csub = &(C[row*32]);
    register unsigned Cm[1000] = {0}; // allocate 1 register <-- can parallelize

    // compute
    #pragma unroll
    for (int i=row_start; i<row_start+workload; i++) {

        unsigned r0 = Asub[(i-row_start)*32+laneid];
        unsigned r1 = Bsub[(colind[i])];

        Cm[i-row_start] += __popc(r0 & r1);
    }

    // store
    T sum = 0;
    for (int i=0; i<workload; i++) sum += (T)(Cm[i]);

    atomicAdd(Csub+laneid, sum); // <-- can tag the continuety


#ifdef PROF
        clock_t stop_time = clock();
        runtime[bx] = (int)(stop_time - start_time);
        load[bx] = workload;
//        GET_LANEID;
//        if (laneid == 1 && load[bx] == 0) {printf("[%d] %d %d\n", bx, (int)(stop_time - start_time), (int)(row_end-row_start));}
#endif
    }
}

// workload split: (1) eliminate 0 workload, (2) split large workload
// workload split, estimate size
__global__ void count_workload_split(int* workloadsize, const int* rowptr, const int nblockrows, const int* colind, const int MAX)
{
    int cnt = 0;
    for(int i=0; i<nblockrows; i++) {
        int load = rowptr[i+1]-rowptr[i];
        if (load == 0) continue;
        if (load > MAX) { // split
            int n = (int)ceilf((float)load / MAX);
            cnt += n;
        } else { // preserve
            cnt += 1;
        }
    }
    *workloadsize = cnt;
}

// workload split, all workload is under MAX
__global__ void workload_split(int* workloadptr, const int* rowptr, const int nblockrows, const int MAX)
{
    int cnt = 0;
    for(int i=0; i<nblockrows; i++) {
        int load = rowptr[i+1]-rowptr[i];

        if (load == 0) continue; // eliminate 0 workload
        if (load > MAX) { // split
            int n = load / MAX;
            int j;
            for(j=0; j<n; j++) {
                workloadptr[cnt++] = i;// row
                workloadptr[cnt++] = rowptr[i]+MAX*j; // rowstart
                workloadptr[cnt++] = MAX; // load
            }

            // rest
            if (load % MAX != 0) {
                workloadptr[cnt++] = i;// row
                workloadptr[cnt++] = rowptr[i]+MAX*j; // rowstart
                workloadptr[cnt++] = load % MAX; // load
            }
        } else { // preserve
            workloadptr[cnt++] = i;// row
            workloadptr[cnt++] = rowptr[i]; // rowstart
            workloadptr[cnt++] = load; // load
        }
    }
} // [row, rowstart, load]

// workload merge: (1) eliminate 0 workload, (2) merge small workload
// workload merge, estimate size
__global__ void count_workload_merge(int* workloadsize, const int* rowptr, const int nblockrows, const int* colind, const int MAX)
{
    int cnt = 0;

    for(int i=0; i<nblockrows; i++) {
        int load = rowptr[i+1]-rowptr[i];
        if (load == 0) continue;
        if (load < MAX) { // merge
            int temp = 0;
            while (temp + load <= MAX) {
                temp += load;
                i += 1;
                load = rowptr[i+1]-rowptr[i];
            }
            cnt += 1;
            i -= 1;
        } else { // preserve
            cnt += 1;
        }
    }
    *workloadsize = cnt;
}

// workload merge, merge until all workload is close to (<=) MAX
__global__ void workload_merge(int* workloadptr, int* rowptr, const int nblockrows, const int* colind, const int MAX)
{
    int cnt = 0;
    for(int i=0; i<nblockrows; i++) {
        int load = rowptr[i+1]-rowptr[i];
        if (load == 0) continue;
        if (load < MAX) { // merge
            workloadptr[cnt++] = i;
            int temp = 0;
            while (temp + load <= MAX) {
                // for this iteration
                temp += load;


                // for next iteration
                i += 1;
                load = rowptr[i+1]-rowptr[i];
                rowptr[i] = -rowptr[i]; // set next
            }
            rowptr[i] = -rowptr[i]; // set back to original
            i -= 1;
        } else { // preserve
            workloadptr[cnt++] = i;
        }
    }
}
// 1 2 3 [4] 6 7 <-- merge workload will revise the nnz boundary on rowptr
// check and found it is merge workload (rowend is -1)
// e.g. rowptr [20 30 30 35 40] => rowptr [20 -30 -30 -35 -40] ...

//======================================================================================
// Considering Workload Balancing -- workload split and merge (evenly distribute)
//======================================================================================
// naive 32
template <typename Index, typename T>
__global__ void bmv32_sparse_workloadmergeNsplit(const unsigned* __restrict__ A, const unsigned* __restrict__ B,
            T* C, const Index* __restrict__ rowptr, const Index* __restrict__ colind,
            const Index* __restrict__ workload_info_list, const Index* __restrict__ workload_size_list_acc,
            const Index workloadsize, const Index MAX,
            const Index nblockrows, const Index nblocks, int* runtime, int* load)
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
//    if (laneid == 0) printf("[%d] numworkload: %d\n", bx, numworkload);
    const unsigned* Bsub = &(B[0]);

    int workload = 0;
    for (int w=0; w<numworkload; w++) {
        // set pointer
        row_start += workload; // move 1 step
        workload = workload_info_list[list_start+2+w]; // get workload
//        if (laneid == 0) printf("[%d] row_start: %d, workload: %d\n", bx, row_start, workload);

        // load location
        const unsigned* Asub = &(A[row_start*32]);
        T* Csub = &(C[(row+w)*32]);

        // compute
        register unsigned Cm[1] = {0}; // allocate 1 register
        for (int i=row_start; i<row_start+workload; i++) {
            unsigned r0 = Asub[(i-row_start)*32+laneid]; // block is in continuous layout
            unsigned r1 = Bsub[(colind[i])]; // only first row is required

            Cm[0] += __popc(r0 & r1);
        }

        // store
        atomicAdd(Csub+laneid, (T)(Cm[0]));
    }

#ifdef PROF
        clock_t stop_time = clock();
        runtime[bx] = (int)(stop_time - start_time);
        load[bx] = row_start + workload - workload_info_list[list_start+1]; // temp
//        GET_LANEID;
//        if (laneid == 1 && load[bx] == 0) {printf("[%d] %d %d\n", bx, (int)(stop_time - start_time), (int)(row_end-row_start));}
#endif
    }
}

// naive 64
template <typename Index, typename T>
__global__ void bmv64_sparse_workloadmergeNsplit(const ullong* __restrict__ A, const ullong* __restrict__ B,
            T* C, const Index* __restrict__ rowptr, const Index* __restrict__ colind,
            const Index* __restrict__ workload_info_list, const Index* __restrict__ workload_size_list_acc,
            const Index workloadsize, const Index MAX,
            const Index nblockrows, const Index nblocks, int* runtime, int* load)
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
    const ullong* Bsub = &(B[0]);

    int workload = 0;
    for (int w=0; w<numworkload; w++) {
        // set pointer
        row_start += workload; // move 1 step
        workload = workload_info_list[list_start+2+w]; // get workload

        // load location
        const ullong* Asub = &(A[row_start*64]);
        T* Csub = &(C[(row+w)*64]); // <-- should not be continuous move, will fail at some case with 0 blockrows (e.g. citpat, indochina)

        // compute
        register unsigned Cm[1] = {0}; // allocate 1 register
        for (int i=row_start; i<row_start+workload; i++) {

            ullong a0 = Asub[(i-row_start)*64+laneid];
            ullong a1 = Asub[(i-row_start)*64+32+laneid];
            ullong b0 = Bsub[colind[i]];

            Cm[0] += (__popcll(a0 & b0) << 16) + __popcll(a1 & b0);
        }

        // store
        short t0, t1;
        asm volatile("mov.b32 {%0,%1}, %2;":"=h"(t1),"=h"(t0):"r"(Cm[0]));
        atomicAdd(Csub+laneid, (T)t0);
        atomicAdd(Csub+laneid+32, (T)t1);
    }

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

//======================================================================================
// Considering Workload Balancing -- BCOO
//======================================================================================
// naive 32
template <typename Index, typename T>
__global__ void bmv32_sparse_opt(const unsigned* __restrict__ A, const unsigned* __restrict__ B, T* C,
                                const Index* __restrict__ rowind, const Index* __restrict__ colind,
                                const Index nblocks, const Index MAX, int* runtime, int* load)
{
    const unsigned bx = blockIdx.x * gridDim.x * gridDim.y + blockIdx.y * gridDim.y + blockIdx.z;

    if (bx < (int)ceil((float)nblocks/MAX)) {
#ifdef PROF
        clock_t start_time = clock();
#endif

    // load
    GET_LANEID;

    // compute // can parallel by max
    int workload_end = bx*MAX+MAX < nblocks ? bx*MAX+MAX : nblocks;
    register unsigned Cm[1] = {0};

    #pragma unroll
    for(int w=bx*MAX; w<workload_end; w++) {
        // set pointer
        int row = rowind[w];
        int col = colind[w];

        // compute
        unsigned r0 = A[w*32+laneid];
        unsigned r1 = B[col];

        // store
        atomicAdd(&C[row*32+laneid], (T)(__popc(r0 & r1)));
    }

#ifdef PROF
        clock_t stop_time = clock();
        runtime[bx] = (int)(stop_time - start_time);
        load[bx] = workload_end-bx*MAX;
#endif
    }
}

// bsr to bcoo
__global__ void bsr2bcoo(const int* rowptr, const int nblockrows, const int* colind, int* rowind) {

    for(int i=0; i<nblockrows; i++) {
        for(int j=rowptr[i]; j<rowptr[i+1]; j++) {
            rowind[j] = i;
        }
    }
}
