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

        if (bx*8+laneid/8<nblockrows) {
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
